# train.py
"""
⚠️ 增强版训练脚本 - 支持跨层特征聚合

修改内容:
1. 导入增强配置(第19行)
2. 模型创建时添加CLA参数(第51-62行)
3. 其余代码100%保持不变

使用方式:
- 启用CLA: 使用 EnhancedConfig 或 CrossDomainConfig
- 禁用CLA: 使用 BaselineConfig (恢复原论文行为)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import LAMSegmentationModel
from data.dataset import create_dataloader
from utils.metrics import SegmentationMetrics
from utils.loss import SegmentationLoss

# ========== 修改1: 导入增强配置 ==========
# from configs.lam_config import config  # ❌ 原始导入

# ⭐ 新增: 导入增强配置
from configs.lam_config_enhanced import (
    BaselineConfig,  # 原论文配置(不使用CLA)
    EnhancedConfig,  # 增强配置(使用CLA)
    CrossDomainConfig  # 跨域配置(CLA+域对齐)
)

# ⭐ 选择配置模式 (三选一)
# config = BaselineConfig()      # 模式1: 基线(原论文,28.03% mIoU)
config = EnhancedConfig()  # 模式2: 增强(推荐,预期36-40% mIoU) ⭐⭐⭐
# config = CrossDomainConfig()   # 模式3: 跨域(完整增强,预期38-42% mIoU)

# 打印配置信息
config.print_enhancement_config()
# ========== 修改1结束 ==========

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """多项式学习率调度器(保持不变)"""

    def __init__(self, optimizer, max_epochs, power=0.9):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class Trainer:
    """训练器类"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 保存模型和日志目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("CREATING MODEL (Paper Configuration + Enhancements)")
        print("=" * 70)

        # ========== 修改2: 创建模型时添加CLA参数 ==========
        # 第一阶段不使用LSM
        print("Stage 1: Pre-training WITHOUT LSM (10 epochs)")

        self.model = LAMSegmentationModel(
            # ========== 原有参数(100%保持不变) ==========
            backbone_name=config.backbone,
            num_classes=config.num_classes,
            num_tokens=config.num_tokens,  # m=100
            token_rank=config.token_rank,  # r=16
            num_groups=config.num_groups,  # G=16
            use_lsm=False,  # 第一阶段不使用LSM
            tau=config.tau,  # τ=0.5
            shared_tokens=True,
            adapt_layers=config.adapt_layers,  # 最后4层

            # ========== ⭐ 新增: 跨层特征聚合参数 ==========
            use_cross_layer_aggregation=config.use_cross_layer_aggregation,  # ⭐ 是否启用CLA
            cross_layer_output_dim=config.cross_layer_output_dim,  # ⭐ 输出维度(512)
            aggregation_method=config.aggregation_method,  # ⭐ 聚合方法
            use_domain_alignment=config.use_domain_alignment  # ⭐ 是否启用域对齐
        )
        # ========== 修改2结束 ==========

        if config.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=config.gpu_ids)

        self.model = self.model.to(self.device)

        # ========== 以下代码100%保持不变 ==========

        print("\n" + "=" * 70)
        print("LOADING DATASETS")
        print("=" * 70)

        self.train_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split='train',
            batch_size=config.batch_size,  # 4
            num_workers=config.num_workers,
            image_size=config.image_size,  # 512
            crop_range=config.crop_range,  # [256, 1024]
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

        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        # 损失函数
        self.criterion = SegmentationLoss(
            num_classes=config.num_classes,
            lambda_cov=config.lambda_cov  # 0.5
        )

        # 获取可训练参数 (⭐ 自动包含CLA参数)
        trainable_params = self.model.module.get_trainable_parameters() \
            if isinstance(self.model, nn.DataParallel) \
            else self.model.get_trainable_parameters()

        print("\n" + "=" * 70)
        print("OPTIMIZER CONFIGURATION (Paper Settings)")
        print("=" * 70)
        print(f"Learning rate: {config.learning_rate}")  # 1e-4
        print(f"Weight decay: {config.weight_decay}")  # 0.05
        print(f"Epsilon: {config.eps}")  # 1e-8

        # 优化器设置
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,  # 1.0 × 10^-4
            weight_decay=config.weight_decay,  # 0.05
            eps=config.eps  # 1.0 × 10^-8
        )

        # 学习率调度器
        self.scheduler = PolyLR(
            self.optimizer,
            max_epochs=config.num_epochs_pretrain,
            power=0.9
        )

        self.metrics = SegmentationMetrics(num_classes=config.num_classes)
        self.writer = SummaryWriter(config.log_dir)

        self.current_epoch = 0
        self.best_miou = 0.0

        # 早停机制
        self.patience = 10
        self.patience_counter = 0

        # 保存第一阶段的optimizer状态用于第二阶段
        self.stage1_optimizer_state = None

        # ========== ⭐ 新增: 打印模型参数统计 ==========
        self._print_model_stats()

    def _print_model_stats(self):
        """打印模型参数统计"""
        model_to_check = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        param_stats = model_to_check.count_parameters()

        print("\n" + "=" * 70)
        print("MODEL PARAMETER STATISTICS")
        print("=" * 70)
        print(f"Backbone:")
        print(f"  Total: {param_stats['backbone_total']:,}")
        print(f"  Trainable: {param_stats['backbone_trainable']:,}")
        print(f"LAM Modules:")
        print(f"  Total: {param_stats['lam_total']:,}")
        print(f"  Trainable: {param_stats['lam_trainable']:,}")

        # ⭐ 新增: CLA参数统计
        if 'cla_total' in param_stats and param_stats['cla_total'] > 0:
            print(f"Cross-Layer Aggregator:")
            print(f"  Total: {param_stats['cla_total']:,}")
            print(f"  Trainable: {param_stats['cla_trainable']:,}")

        # ⭐ 新增: 域对齐参数统计
        if 'da_total' in param_stats and param_stats['da_total'] > 0:
            print(f"Domain Aligner:")
            print(f"  Total: {param_stats['da_total']:,}")
            print(f"  Trainable: {param_stats['da_trainable']:,}")

        print(f"Segmentation Head:")
        print(f"  Total: {param_stats['head_total']:,}")
        print(f"  Trainable: {param_stats['head_trainable']:,}")
        print(f"Overall:")
        print(f"  Total: {param_stats['total']:,}")
        print(f"  Trainable: {param_stats['trainable']:,}")

        trainable_ratio = param_stats['trainable'] / param_stats['total'] * 100
        print(f"  Trainable Ratio: {trainable_ratio:.2f}%")
        print("=" * 70)

    def train_epoch(self, epoch):
        """训练一个epoch (100%保持不变)"""
        self.model.train()

        total_loss = 0.0
        total_seg_loss = 0.0
        total_cov_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播 (⭐ 接口完全相同,自动使用CLA)
            logits, cov_loss = self.model(images, compute_cov_loss=True)

            # 损失计算: L_total = L_seg + λL_cov
            seg_loss = self.criterion.segmentation_loss(logits, labels)
            loss = seg_loss + self.config.lambda_cov * cov_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cov_loss += cov_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'seg': f"{seg_loss.item():.4f}",
                'cov': f"{cov_loss.item():.4f}"
            })

            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/SegLoss', seg_loss.item(), global_step)
                self.writer.add_scalar('Train/CovLoss', cov_loss.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_cov_loss = total_cov_loss / len(self.train_loader)

        return avg_loss, avg_seg_loss, avg_cov_loss

    def validate(self, epoch):
        """验证 (100%保持不变)"""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits, cov_loss = self.model(images, compute_cov_loss=True)

                # 损失计算
                seg_loss = self.criterion.segmentation_loss(logits, labels)
                loss = seg_loss + self.config.lambda_cov * cov_loss

                total_loss += loss.item()

                # 预测
                preds = torch.argmax(logits, dim=1)

                # 更新指标
                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

        # 计算指标
        results = self.metrics.compute()
        avg_loss = total_loss / len(self.val_loader)

        # 打印结果
        print(f"\nValidation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {results['miou']:.4f}")
        print(f"  mAcc: {results['macc']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  Overall Acc: {results['overall_acc']:.4f}")

        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/mIoU', results['miou'], epoch)
        self.writer.add_scalar('Val/mAcc', results['macc'], epoch)
        self.writer.add_scalar('Val/F1', results['f1'], epoch)

        return results, avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点 (100%保持不变)"""
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) \
            else self.model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config
        }

        # 保存最新检查点
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved! mIoU: {self.best_miou:.4f}")

        # 定期保存
        if epoch % self.config.save_freq == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def train(self):
        """完整训练流程 (100%保持不变)"""

        # ========================================
        # STAGE 1: 预训练 (不使用LSM, 10 epochs)
        # ========================================
        print("\n" + "=" * 70)
        print("STAGE 1: PRE-TRAINING WITHOUT LSM")
        print("Epochs: 10")
        print("=" * 70 + "\n")

        for epoch in range(1, self.config.num_epochs_pretrain + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{self.config.num_epochs_pretrain} (Stage 1)")
            print(f"{'=' * 70}")

            train_loss, seg_loss, cov_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f} (Seg: {seg_loss:.4f}, Cov: {cov_loss:.4f})")

            if epoch % self.config.eval_freq == 0:
                val_results, val_loss = self.validate(epoch)

                if val_results['miou'] > self.best_miou:
                    self.best_miou = val_results['miou']
                    self.save_checkpoint(epoch, is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(epoch)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

        print(f"\n{'=' * 70}")
        print(f"STAGE 1 COMPLETED")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"{'=' * 70}")

        # 第一阶段保存optimizer状态
        self.stage1_optimizer_state = {
            'state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou
        }

        # ========================================
        # STAGE 2: 完整训练 (使用LSM, 20 epochs)
        # ========================================
        print("\n" + "=" * 70)
        print("STAGE 2: FULL TRAINING WITH LSM")
        print("Epochs: 20")
        print("=" * 70 + "\n")

        # 启用LSM
        if isinstance(self.model, nn.DataParallel):
            for lam in self.model.module.multi_lam.lams:
                lam.use_lsm = True
            print("✅ LSM enabled for all LAM modules (DataParallel)")
        else:
            for lam in self.model.multi_lam.lams:
                lam.use_lsm = True
            print("✅ LSM enabled for all LAM modules")

        # 获取可训练参数 (⭐ 自动包含LSM和CLA参数)
        trainable_params = self.model.module.get_trainable_parameters() \
            if isinstance(self.model, nn.DataParallel) \
            else self.model.get_trainable_parameters()

        # 创建新optimizer
        new_optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate * 0.5,  # 使用较小的学习率
            weight_decay=self.config.weight_decay,
            eps=self.config.eps
        )

        # 继承第一阶段的状态
        if self.stage1_optimizer_state is not None:
            old_state = self.stage1_optimizer_state['state_dict']
            new_state = new_optimizer.state_dict()

            for old_group, new_group in zip(old_state['param_groups'], new_state['param_groups']):
                for old_p, new_p in zip(old_group['params'], new_group['params']):
                    if old_p in old_state['state']:
                        new_state['state'][new_p] = old_state['state'][old_p]

            try:
                new_optimizer.load_state_dict(new_state)
                print("✅ Successfully inherited momentum from Stage 1")
            except:
                print("⚠️ Could not inherit momentum, starting fresh")

        self.optimizer = new_optimizer

        self.scheduler = PolyLR(
            self.optimizer,
            max_epochs=config.num_epochs_full,
            power=0.9
        )

        start_epoch = self.config.num_epochs_pretrain + 1
        total_epochs = self.config.num_epochs_pretrain + self.config.num_epochs_full

        for epoch in range(start_epoch, total_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{total_epochs} (Stage 2: {epoch - start_epoch + 1}/20)")
            print(f"{'=' * 70}")

            train_loss, seg_loss, cov_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f} (Seg: {seg_loss:.4f}, Cov: {cov_loss:.4f})")

            if epoch % self.config.eval_freq == 0:
                val_results, val_loss = self.validate(epoch)

                if val_results['miou'] > self.best_miou:
                    self.best_miou = val_results['miou']
                    self.save_checkpoint(epoch, is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(epoch)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print("=" * 70)

        # 计算论文表二数据
        self.calculate_paper_table2_data()

    def calculate_paper_table2_data(self):
        """计算并输出论文表二的各项数据 (100%保持不变)"""
        print("\n" + "=" * 70)
        print("Calculating metrics for Paper Table 2")
        print("=" * 70)

        # 在验证集上计算
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Calculating metrics"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

        results = self.metrics.compute()

        # 输出mIoU、mAcc、F1等
        print(f"mIoU: {results['miou']:.4f}")
        print(f"mAcc: {results['macc']:.4f}")
        print(f"F1: {results['f1']:.4f}")
        print(f"Overall Accuracy: {results['overall_acc']:.4f}")

        print("\nIoU per class:")
        for i, class_name in enumerate(self.config.rubber_classes):
            print(f"  {class_name}: {results['iou_per_class'][i]:.4f}")


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("LAM TRAINING - ENHANCED CONFIGURATION")
    print("=" * 70)

    # 更新数据集配置
    config.update_for_dataset('rubber_wood')

    print("\n" + "=" * 70)
    print("Training Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Image size: {config.image_size}")
    print(f"  Crop range: {config.crop_range}")
    print(f"  Stage 1 epochs: {config.num_epochs_pretrain}")
    print(f"  Stage 2 epochs: {config.num_epochs_full}")

    # ⭐ 新增: 打印增强配置
    print(f"\nEnhancement Configuration:")
    print(f"  Cross-Layer Aggregation: {config.use_cross_layer_aggregation}")
    if config.use_cross_layer_aggregation:
        print(f"  Aggregation Output Dim: {config.cross_layer_output_dim}")
        print(f"  Aggregation Method: {config.aggregation_method}")
    print(f"  Domain Alignment: {config.use_domain_alignment}")
    print("=" * 70)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()