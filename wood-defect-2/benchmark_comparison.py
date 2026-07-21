# benchmark_comparison.py
"""
对比模型训练和评估脚本
实现论文Table 2中的所有对比模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_dataloader
from utils.metrics import SegmentationMetrics
from configs.lam_config import config


# ========================================
# 1. 模型定义
# ========================================

class ModelFactory:
    """模型工厂 - 创建论文中的对比模型"""

    @staticmethod
    def create_model(model_name: str, num_classes: int, pretrained: bool = True):
        """
        创建模型
        Args:
            model_name: 模型名称
            num_classes: 类别数
            pretrained: 是否使用预训练权重
        """
        import segmentation_models_pytorch as smp

        if model_name == "unet":
            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes
            )

        elif model_name == "fcn":
            from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
            import torch.nn as nn

            if pretrained:
                # 确保使用正确的预训练配置
                model = fcn_resnet50(
                    weights=FCN_ResNet50_Weights.DEFAULT,
                    aux_loss=True  # 确保包含辅助分类器
                )

                # 调整分类头为6个类别
                in_channels = model.classifier[4].in_channels
                model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

                # 调整辅助分类器
                if model.aux_classifier is not None:
                    aux_in_channels = model.aux_classifier[4].in_channels
                    model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
            else:
                model = fcn_resnet50(
                    weights=None,
                    num_classes=num_classes,
                    aux_loss=True  # 确保包含辅助分类器
                )

        elif model_name == "deeplabv3":
            model = smp.DeepLabV3(
                encoder_name="resnet50",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes
            )

        elif model_name == "deeplabv3plus":
            model = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes
            )

        elif model_name == "segformer":
            # 需要安装: pip install mmsegmentation
            try:
                from mmseg.models import build_segmentor
                cfg = dict(
                    type='EncoderDecoder',
                    backbone=dict(
                        type='MixVisionTransformer',
                        in_channels=3,
                        embed_dims=64,
                        num_stages=4,
                        num_layers=[2, 2, 2, 2],
                        num_heads=[1, 2, 5, 8],
                        patch_sizes=[7, 3, 3, 3],
                        sr_ratios=[8, 4, 2, 1],
                        out_indices=(0, 1, 2, 3),
                        pretrained=None
                    ),
                    decode_head=dict(
                        type='SegformerHead',
                        in_channels=[64, 128, 320, 512],
                        in_index=[0, 1, 2, 3],
                        channels=256,
                        dropout_ratio=0.1,
                        num_classes=num_classes,
                        align_corners=False
                    )
                )
                model = build_segmentor(cfg)
            except:
                print("⚠️ SegFormer需要mmsegmentation,使用DeepLabV3+代替")
                model = ModelFactory.create_model("deeplabv3plus", num_classes, pretrained)

        else:
            raise ValueError(f"不支持的模型: {model_name}")

        return model


# ========================================
# 2. 训练器
# ========================================

class BenchmarkTrainer:
    """基准模型训练器"""

    def __init__(self, model_name: str, config):
        self.model_name = model_name
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 创建模型
        print(f"\n{'=' * 70}")
        print(f"创建模型: {model_name}")
        print(f"{'=' * 70}")

        self.model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=config.num_classes,
            pretrained=True
        )
        self.model = self.model.to(self.device)

        # 数据加载器
        self.train_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split='train',
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            crop_range=config.crop_range,
            augmentation=True
        )

        self.val_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split='val',
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            augmentation=False
        )

        # 损失和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=config.num_epochs_full,
            power=0.9
        )

        self.metrics = SegmentationMetrics(num_classes=config.num_classes)
        self.best_miou = 0.0

        # 输出目录
        self.output_dir = os.path.join(config.output_root, f'benchmark_{model_name}')
        os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            if self.model_name == "fcn":
                outputs = self.model(images)['out']
            else:
                outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                if self.model_name == "fcn":
                    outputs = self.model(images)['out']
                else:
                    outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        results = self.metrics.compute()
        avg_loss = total_loss / len(self.val_loader)

        print(f"\nValidation Results (Epoch {epoch}):")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {results['miou']:.4f}")
        print(f"  mAcc: {results['macc']:.4f}")
        print(f"  F1: {results['f1']:.4f}")

        return results, avg_loss

    def train(self, num_epochs: int = 30):
        """完整训练流程"""
        print(f"\n{'=' * 70}")
        print(f"开始训练 {self.model_name}")
        print(f"总epochs: {num_epochs}")
        print(f"{'=' * 70}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 70}")

            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            if epoch % 10 == 0:
                val_results, val_loss = self.validate(epoch)

                if val_results['miou'] > self.best_miou:
                    self.best_miou = val_results['miou']
                    self.save_checkpoint(epoch, val_results, is_best=True)

            self.scheduler.step()

        print(f"\n{'=' * 70}")
        print(f"{self.model_name} 训练完成!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"{'=' * 70}\n")

        return self.best_miou

    def save_checkpoint(self, epoch: int, results: dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'results': results
        }

        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ 保存最佳模型: {best_path}")


# ========================================
# 3. 批量评估
# ========================================

class BenchmarkEvaluator:
    """批量评估所有模型"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 论文中的对比模型
        self.models = [
            "unet",
            "fcn",
            "deeplabv3",
            "deeplabv3plus",
            "segformer",  # 可选,需要额外依赖
        ]

        self.results_table = []

    def train_all_models(self, num_epochs: int = 30):
        """训练所有模型"""
        print(f"\n{'=' * 70}")
        print("开始批量训练所有对比模型")
        print(f"模型列表: {self.models}")
        print(f"{'=' * 70}\n")

        for model_name in self.models:
            try:
                trainer = BenchmarkTrainer(model_name, self.config)
                best_miou = trainer.train(num_epochs)

                # 加载最佳模型并评估
                results = self.evaluate_model(model_name)
                self.results_table.append(results)

            except Exception as e:
                print(f"❌ {model_name} 训练失败: {e}")
                continue

        # self.save_results_table()

    def evaluate_model(self, model_name: str) -> Dict:
        """评估单个模型"""
        print(f"\n{'=' * 70}")
        print(f"评估模型: {model_name}")
        print(f"{'=' * 70}")

        # 加载最佳模型
        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=self.config.num_classes,
            pretrained=False
        )

        checkpoint_path = os.path.join(
            self.config.output_root,
            f'benchmark_{model_name}',
            'best_model.pth'
        )

        if not os.path.exists(checkpoint_path):
            print(f"❌ 未找到模型: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # 评估
        val_loader = create_dataloader(
            root_dir=self.config.rubber_wood_path,
            split='val',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            image_size=self.config.image_size,
            augmentation=False
        )

        metrics = SegmentationMetrics(num_classes=self.config.num_classes)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                if model_name == "fcn":
                    outputs = model(images)['out']
                else:
                    outputs = model(images)

                preds = torch.argmax(outputs, dim=1)
                metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

        results = metrics.compute()

        # 构建结果字典
        result_dict = {
            'Method': model_name.upper(),
            'BG': results['iou_per_class'][0] * 100,
            'SK': results['iou_per_class'][1] * 100,
            'DK': results['iou_per_class'][2] * 100,
            'CK': results['iou_per_class'][3] * 100,
            'ME': results['iou_per_class'][4] * 100,
            'TC': results['iou_per_class'][5] * 100,
            'mIoU': results['miou'] * 100,
            'mACC': results['macc'] * 100,
            'F1': results['f1'] * 100
        }

        print(f"\n{model_name} 结果:")
        print(f"  mIoU: {result_dict['mIoU']:.2f}%")
        print(f"  mACC: {result_dict['mACC']:.2f}%")
        print(f"  F1: {result_dict['F1']:.2f}%")

        return result_dict

    def save_results_table(self):
        """保存结果表格"""
        if not self.results_table:
            print("⚠️ 没有结果可保存")
            return

        # 创建DataFrame
        df = pd.DataFrame(self.results_table)

        # 保存为CSV
        csv_path = os.path.join(self.config.output_root, 'deeplabv3.csv')
        df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"\n✅ 结果已保存: {csv_path}")

        # 打印表格
        print(f"\n{'=' * 70}")
        print("所有模型评估结果 (类似Table 2)")
        print(f"{'=' * 70}\n")
        print(df.to_string(index=False))
        print(f"\n{'=' * 70}\n")

        # 保存为LaTeX格式
        latex_path = os.path.join(self.config.output_root, 'result.tex')
        df.to_latex(latex_path, index=False, float_format='%.2f')
        print(f"✅ LaTeX表格已保存: {latex_path}")


# ========================================
# 4. 主函数
# ========================================

def main():
    """主函数"""
    print(f"\n{'=' * 70}")
    print("对比模型训练和评估")
    print(f"{'=' * 70}\n")

    # 更新配置
    config.update_for_dataset('rubber_wood')
    config.create_output_dirs()

    # 创建评估器
    evaluator = BenchmarkEvaluator(config)

    # 选项1: 训练所有模型
    print("\n请选择操作:")
    print("1. 训练所有对比模型 (耗时较长)")
    print("2. 仅评估已训练的模型")
    print("3. 训练单个模型")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == "1":
        num_epochs = int(input("请输入训练epochs (建议30): ").strip() or "30")
        evaluator.train_all_models(num_epochs)

    elif choice == "2":
        # 评估所有模型
        for model_name in evaluator.models:
            try:
                results = evaluator.evaluate_model(model_name)
                if results:
                    evaluator.results_table.append(results)
            except Exception as e:
                print(f"❌ {model_name} 评估失败: {e}")

        # evaluator.save_results_table()

    elif choice == "3":
        print("\n可用模型:")
        for i, model_name in enumerate(evaluator.models, 1):
            print(f"{i}. {model_name}")

        model_idx = int(input("\n请选择模型编号: ").strip()) - 1
        if 0 <= model_idx < len(evaluator.models):
            model_name = evaluator.models[model_idx]
            num_epochs = int(input(f"请输入训练epochs (建议30): ").strip() or "30")

            trainer = BenchmarkTrainer(model_name, config)
            trainer.train(num_epochs)

            results = evaluator.evaluate_model(model_name)
            if results:
                evaluator.results_table.append(results)
                evaluator.save_results_table()

    else:
        print("❌ 无效选项")


if __name__ == "__main__":
    main()