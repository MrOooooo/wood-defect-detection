# train_mask2former.py
"""
LAM + Mask2Former 训练脚本

严格按照论文两阶段训练策略:
- Stage 1: 预训练10 epochs, 不使用LSM
- Stage 2: 完整训练20 epochs, 使用LSM

关键改进:
- 使用Mask2Former Decoder替代简单Conv Head
- 修复lambda_cov为0.5 (论文设置)
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

from models.lam_mask2former import LAMMask2FormerModel
from data.dataset import create_dataloader
from utils.metrics import SegmentationMetrics
from configs.lam_config import config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Mask2FormerLoss(nn.Module):
    """
    Mask2Former风格的损失函数
    包含:
    - Cross-entropy loss for class prediction
    - Binary cross-entropy + Dice loss for mask prediction
    - Covariance regularization loss from FDM
    """

    def __init__(self, num_classes, lambda_cov=0.5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cov = lambda_cov
        self.ignore_index = ignore_index

        # 类别损失权重 (可根据类别不平衡调整)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def dice_loss(self, pred, target, smooth=1e-5):
        """
        Dice Loss
        Args:
            pred: (B, C, H, W) 预测概率
            target: (B, H, W) 目标标签
        """
        # 创建有效mask
        valid_mask = (target != self.ignore_index)

        # One-hot编码
        target_clamped = target.clone()
        target_clamped[~valid_mask] = 0
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target_clamped.unsqueeze(1), 1)

        # 只在有效区域计算
        valid_mask = valid_mask.unsqueeze(1).expand_as(pred)
        pred = pred * valid_mask.float()
        target_one_hot = target_one_hot * valid_mask.float()

        # Dice计算
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss

    def forward(self, logits, labels):
        """
        计算分割损失
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W)
        Returns:
            loss: 总损失
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)

        # Dice loss
        probs = torch.softmax(logits, dim=1)
        dice_loss = self.dice_loss(probs, labels)

        # 总损失
        loss = ce_loss + dice_loss

        return loss


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial学习率衰减 (论文使用)"""

    def __init__(self, optimizer, max_epochs, power=0.9):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class Mask2FormerTrainer:
    """LAM + Mask2Former 训练器"""

    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("LAM + Mask2Former Training (Paper Configuration)")
        print("=" * 70)

        # ========== 创建模型 ==========
        print("\nStage 1: Pre-training WITHOUT LSM (10 epochs)")
        self.model = LAMMask2FormerModel(
            backbone_name=cfg.backbone,
            num_classes=cfg.num_classes,
            # LAM参数 (论文设置)
            num_tokens=cfg.num_tokens,  # m=100
            token_rank=cfg.token_rank,  # r=16
            num_groups=cfg.num_groups,  # G=16
            use_lsm=False,  # 第一阶段不使用LSM
            tau=cfg.tau,  # τ=0.5
            shared_tokens=True,
            adapt_layers=cfg.adapt_layers,  # [8,9,10,11]
            # Mask2Former参数
            hidden_dim=256,
            num_queries=100,
            num_decoder_layers=6,
            num_heads=8
        )

        if cfg.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=cfg.gpu_ids)

        self.model = self.model.to(self.device)

        # ========== 数据加载 ==========
        print("\n" + "=" * 70)
        print("Loading Datasets")
        print("=" * 70)

        self.train_loader = create_dataloader(
            root_dir=cfg.rubber_wood_path,
            split='train',
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
            crop_range=cfg.crop_range,
            augmentation=True,
            num_classes=cfg.num_classes
        )

        self.val_loader = create_dataloader(
            root_dir=cfg.rubber_wood_path,
            split='val',
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
            augmentation=False,
            num_classes=cfg.num_classes
        )

        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        # ========== 损失函数 ==========
        # 论文: λ = 0.5
        self.lambda_cov = 0.5  # 修正为论文值
        self.criterion = Mask2FormerLoss(
            num_classes=cfg.num_classes,
            lambda_cov=self.lambda_cov,
            ignore_index=cfg.ignore_index
        )

        # ========== 优化器 ==========
        trainable_params = self.model.module.get_trainable_parameters() \
            if isinstance(self.model, nn.DataParallel) \
            else self.model.get_trainable_parameters()

        print("\n" + "=" * 70)
        print("Optimizer Configuration (Paper Settings)")
        print("=" * 70)
        print(f"Learning rate: {cfg.learning_rate}")
        print(f"Weight decay: {cfg.weight_decay}")
        print(f"Lambda_cov: {self.lambda_cov}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            eps=cfg.eps
        )

        self.scheduler = PolyLR(
            self.optimizer,
            max_epochs=cfg.num_epochs_pretrain,
            power=0.9
        )

        # ========== 评估和日志 ==========
        self.metrics = SegmentationMetrics(num_classes=cfg.num_classes)
        self.writer = SummaryWriter(os.path.join(cfg.log_dir, 'mask2former'))

        self.current_epoch = 0
        self.best_miou = 0.0
        self.patience = 15
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_seg_loss = 0.0
        total_cov_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            logits, cov_loss = self.model(images, compute_cov_loss=True)

            # 分割损失
            seg_loss = self.criterion(logits, labels)

            # 总损失: L = L_seg + λ * L_cov
            loss = seg_loss + self.lambda_cov * cov_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

            # TensorBoard
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
        """验证"""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(images, compute_cov_loss=False)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        results = self.metrics.compute()
        avg_loss = total_loss / len(self.val_loader)

        # TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/mIoU', results['miou'], epoch)
        self.writer.add_scalar('Val/mAcc', results['macc'], epoch)
        self.writer.add_scalar('Val/F1', results['f1'], epoch)

        print(f"\nValidation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {results['miou']:.4f}")
        print(f"  mAcc: {results['macc']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  IoU per class: {results['iou_per_class']}")

        return results, avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        model_state = self.model.module.state_dict() \
            if isinstance(self.model, nn.DataParallel) \
            else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'model_type': 'LAMMask2FormerModel'
        }

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'mask2former_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'mask2former_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def train(self):
        """完整训练流程 - 两阶段策略"""

        # ========================================
        # STAGE 1: 预训练 (不使用LSM, 10 epochs)
        # ========================================
        print("\n" + "=" * 70)
        print("STAGE 1: PRE-TRAINING WITHOUT LSM")
        print(f"Epochs: {self.config.num_epochs_pretrain}")
        print(f"Lambda_cov: {self.lambda_cov}")
        print("=" * 70 + "\n")

        for epoch in range(1, self.config.num_epochs_pretrain + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{self.config.num_epochs_pretrain}")
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

            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(epoch)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

        print(f"\n{'=' * 70}")
        print(f"STAGE 1 COMPLETED")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"{'=' * 70}")

        # ========================================
        # STAGE 2: 完整训练 (使用LSM, 20 epochs)
        # ========================================
        print("\n" + "=" * 70)
        print("STAGE 2: FULL TRAINING WITH LSM")
        print(f"Epochs: {self.config.num_epochs_full}")
        print("=" * 70 + "\n")

        # 启用LSM
        if isinstance(self.model, nn.DataParallel):
            for lam in self.model.module.multi_lam.lams:
                lam.use_lsm = True
        else:
            for lam in self.model.multi_lam.lams:
                lam.use_lsm = True
        print("✅ LSM enabled for all LAM modules")

        # 重置优化器 (降低学习率)
        trainable_params = self.model.module.get_trainable_parameters() \
            if isinstance(self.model, nn.DataParallel) \
            else self.model.get_trainable_parameters()

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate * 0.5,  # 降低学习率
            weight_decay=self.config.weight_decay,
            eps=self.config.eps
        )

        self.scheduler = PolyLR(
            self.optimizer,
            max_epochs=self.config.num_epochs_full,
            power=0.9
        )

        self.patience_counter = 0

        start_epoch = self.config.num_epochs_pretrain + 1
        total_epochs = self.config.num_epochs_pretrain + self.config.num_epochs_full

        for epoch in range(start_epoch, total_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{total_epochs} (Stage 2: {epoch - start_epoch + 1}/{self.config.num_epochs_full})")
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

        # 输出最终评估
        self.final_evaluation()

    def final_evaluation(self):
        """最终评估 - 输出论文Table 2格式的数据"""
        print("\n" + "=" * 70)
        print("Final Evaluation (Paper Table 2 Format)")
        print("=" * 70)

        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Final Evaluation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

        results = self.metrics.compute()

        print(f"\n{'=' * 50}")
        print("RESULTS (Rubber Wood Dataset)")
        print(f"{'=' * 50}")
        print(f"mIoU:  {results['miou'] * 100:.2f}%")
        print(f"mAcc:  {results['macc'] * 100:.2f}%")
        print(f"F1:    {results['f1'] * 100:.2f}%")
        print(f"\nPer-class IoU:")

        class_names = ['BG', 'DK', 'SK', 'ME', 'TC', 'CK']
        for i, (name, iou) in enumerate(zip(class_names, results['iou_per_class'])):
            print(f"  {name}: {iou * 100:.2f}%")

        print(f"{'=' * 50}")


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("LAM + Mask2Former Training")
    print("Paper: Efficient Adaptation of Visual Foundation Models")
    print("       for Wood Defect Segmentation")
    print("=" * 70)

    # 更新配置
    config.update_for_dataset('rubber_wood')

    print("\nConfiguration:")
    print(f"  Dataset: Rubber Wood")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Image size: {config.image_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Stage 1 epochs: {config.num_epochs_pretrain}")
    print(f"  Stage 2 epochs: {config.num_epochs_full}")
    print(f"  LAM: m={config.num_tokens}, r={config.token_rank}, G={config.num_groups}")

    trainer = Mask2FormerTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
