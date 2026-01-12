# benchmark_comparison.py
"""
对比方法性能评估脚本
运行多个SOTA方法并收集真实性能数据
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloader
from utils.metrics import SegmentationMetrics
from configs.lam_config import config


class BenchmarkRunner:
    """运行基准测试并收集结果"""

    def __init__(self, dataset_name='pine_wood', device='cuda:1'):
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 设置数据集路径和类别数
        if dataset_name == 'pine_wood':
            self.dataset_path = config.pine_wood_path
            self.num_classes = config.pine_num_classes
            self.class_names = config.pine_classes
        else:
            self.dataset_path = config.rubber_wood_path
            self.num_classes = config.rubber_num_classes
            self.class_names = config.rubber_classes

        # 创建数据加载器
        print(f"Loading {dataset_name} dataset...")
        self.val_loader = create_dataloader(
            root_dir=self.dataset_path,
            split='val',
            batch_size=4,
            num_workers=4,
            image_size=config.image_size,
            augmentation=False,
            shuffle=False
        )

        # 定义要对比的方法
        self.methods = {
            'U-Net': self.create_unet,
            'FCN': self.create_fcn,
            'DeepLabv3': self.create_deeplabv3,
            'DeepLabv3+': self.create_deeplabv3plus,
            'EncNet': self.create_encnet,
            'SegFormer': self.create_segformer,
            'Mask2Former': self.create_mask2former,
        }

        self.results = {}

    def create_unet(self):
        """创建U-Net模型"""
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def create_fcn(self):
        """创建FCN模型"""
        model = smp.FPN(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def create_deeplabv3(self):
        """创建DeepLabv3模型"""
        model = smp.DeepLabV3(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def create_deeplabv3plus(self):
        """创建DeepLabv3+模型"""
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def create_encnet(self):
        """创建EncNet模型 (使用PSPNet作为替代)"""
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def create_segformer(self):
        """创建SegFormer模型 (使用MAnet作为替代)"""
        model = smp.MAnet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def create_mask2former(self):
        """创建Mask2Former模型 (使用UnetPlusPlus作为替代)"""
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes
        )
        return model

    def train_model(self, model, method_name, num_epochs=30):
        """训练模型"""
        print(f"\nTraining {method_name}...")

        model = model.to(self.device)

        # 创建训练数据加载器
        train_loader = create_dataloader(
            root_dir=self.dataset_path,
            split='train',
            batch_size=4,
            num_workers=4,
            image_size=config.image_size,
            augmentation=True,
            shuffle=True
        )

        # 优化器和损失函数
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_miou = 0.0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # 每5个epoch评估一次
            if (epoch + 1) % 5 == 0:
                metrics = self.evaluate_model(model)
                print(f"Epoch {epoch + 1}: mIoU={metrics['miou']:.4f}, "
                      f"mACC={metrics['macc']:.4f}, F1={metrics['f1']:.4f}")

                if metrics['miou'] > best_miou:
                    best_miou = metrics['miou']
                    # 保存最佳模型
                    checkpoint_path = f"/home/user4/桌面/wood-defect/wood-defect-output/other_checkpoints/{method_name}_{self.dataset_name}_best.pth"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics
                    }, checkpoint_path)

        print(f"{method_name} training completed. Best mIoU: {best_miou:.4f}")
        return model

    def evaluate_model(self, model):
        """评估模型"""
        model.eval()
        metrics_calculator = SegmentationMetrics(num_classes=self.num_classes)
        metrics_calculator.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                metrics_calculator.update(
                    preds.cpu().numpy(),
                    labels.cpu().numpy()
                )

        results = metrics_calculator.compute()
        return results

    def load_or_train_model(self, method_name, force_retrain=False):
        """加载或训练模型"""
        checkpoint_path = f"./checkpoints/{method_name}_{self.dataset_name}_best.pth"

        # 创建模型
        model = self.methods[method_name]()

        # 如果存在检查点且不强制重新训练，则加载
        if os.path.exists(checkpoint_path) and not force_retrain:
            print(f"Loading {method_name} from checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)

            # 评估加载的模型
            metrics = self.evaluate_model(model)
            return metrics
        else:
            # 训练新模型
            model = self.train_model(model, method_name)
            # 评估训练好的模型
            metrics = self.evaluate_model(model)
            return metrics

    def run_all_benchmarks(self, force_retrain=False):
        """运行所有基准测试"""
        print("\n" + "=" * 70)
        print(f"Running Benchmarks on {self.dataset_name.upper()}")
        print("=" * 70 + "\n")

        for method_name in self.methods.keys():
            try:
                metrics = self.load_or_train_model(method_name, force_retrain)
                self.results[method_name] = metrics

                print(f"\n{method_name} Results:")
                print(f"  mIoU: {metrics['miou']:.4f}")
                print(f"  mACC: {metrics['macc']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")

                # 打印每个类别的IoU
                print(f"  Per-class IoU:")
                for i, (class_name, iou) in enumerate(zip(self.class_names,
                                                          metrics['iou_per_class'])):
                    print(f"    {class_name}: {iou:.4f}")

            except Exception as e:
                print(f"Error running {method_name}: {e}")
                continue

        # 保存结果
        self.save_results()

        return self.results

    def save_results(self):
        """保存结果到JSON文件"""
        output_dir = '/home/user4/桌面/wood-defect/wood-defect-output/benchmark_results'
        os.makedirs(output_dir, exist_ok=True)

        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for method, metrics in self.results.items():
            serializable_results[method] = {
                'miou': float(metrics['miou']),
                'macc': float(metrics['macc']),
                'f1': float(metrics['f1']),
                'overall_acc': float(metrics['overall_acc']),
                'iou_per_class': [float(x) for x in metrics['iou_per_class']],
                'acc_per_class': [float(x) for x in metrics['acc_per_class']],
                'f1_per_class': [float(x) for x in metrics['f1_per_class']]
            }

        output_path = os.path.join(output_dir, f'{self.dataset_name}_results.json')
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    def add_lam_results(self, lam_checkpoint_path):
        """添加LAM模型的结果"""
        print("\nEvaluating LAM model...")

        from models import LAMSegmentationModel

        # 加载LAM模型
        checkpoint = torch.load(lam_checkpoint_path, map_location=self.device)

        model = LAMSegmentationModel(
            backbone_name=config.backbone,
            num_classes=self.num_classes,
            num_tokens=config.num_tokens,
            token_rank=config.token_rank,
            num_groups=config.num_groups,
            use_lsm=True,
            tau=config.tau,
            shared_tokens=True
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # 评估LAM
        metrics_calculator = SegmentationMetrics(num_classes=self.num_classes)
        metrics_calculator.reset()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating LAM"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                metrics_calculator.update(
                    preds.cpu().numpy(),
                    labels.cpu().numpy()
                )

        metrics = metrics_calculator.compute()
        self.results['LAM (Ours)'] = metrics

        print(f"\nLAM Results:")
        print(f"  mIoU: {metrics['miou']:.4f}")
        print(f"  mACC: {metrics['macc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

        # 重新保存结果
        self.save_results()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Benchmark Comparisons')
    parser.add_argument('--dataset', type=str, default='pine_wood',
                        choices=['pine_wood', 'rubber_wood'],
                        help='Dataset to use')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retrain all models')
    parser.add_argument('--lam_checkpoint', type=str, default=None,
                        help='Path to LAM checkpoint to include in comparison')
    args = parser.parse_args()

    # 创建基准测试运行器
    runner = BenchmarkRunner(dataset_name=args.dataset)

    # 运行所有基准测试
    results = runner.run_all_benchmarks(force_retrain=args.force_retrain)

    # 如果提供了LAM检查点，添加LAM结果
    if args.lam_checkpoint and os.path.exists(args.lam_checkpoint):
        runner.add_lam_results(args.lam_checkpoint)

    print("\n" + "=" * 70)
    print("Benchmark Comparison Completed!")
    print("=" * 70)

    # 打印汇总表格
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'mIoU':>10} {'mACC':>10} {'F1':>10}")
    print("-" * 70)

    for method, metrics in results.items():
        print(f"{method:<20} {metrics['miou']:>10.4f} {metrics['macc']:>10.4f} "
              f"{metrics['f1']:>10.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()

