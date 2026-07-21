# pretrain_lsfa.py
"""
LSFA预训练脚本
使用无标注木材数据进行自监督预训练
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.backbone import VFMBackbone
from test_lsfa.lsfa_model import LSFAPretrainer, WoodTextureAugmentation


class UnlabeledWoodDataset(Dataset):
    """
    无标注木材数据集
    用于LSFA自监督预训练
    """

    def __init__(self, root_dir, image_size=512):
        self.root_dir = root_dir
        self.image_size = image_size

        # 收集所有图像
        self.image_paths = []
        image_dir = os.path.join(root_dir, 'JPEGImages')

        if os.path.exists(image_dir):
            for fname in os.listdir(image_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(image_dir, fname))

        # 数据增强
        self.augmentation = WoodTextureAugmentation()

        print(f"📂 Loaded {len(self.image_paths)} unlabeled images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # 生成两个增强版本
        view1, view2 = self.augmentation(image)

        return {
            'view1': view1,
            'view2': view2,
            'filename': os.path.basename(img_path)
        }


class LSFATrainer:
    """LSFA预训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        config.create_output_dirs()

        print("\n" + "=" * 70)
        print("LSFA PRETRAINING SETUP")
        print("=" * 70)

        # ========== 加载VFM Backbone ==========
        print("\n📦 Loading DINOv2 backbone...")
        self.backbone = VFMBackbone(
            model_name='dinov2',
            freeze=True,
            output_layers=config.adapt_layers
        ).to(self.device)

        # ========== 创建LSFA预训练器 ==========
        print("\n🔧 Creating LSFA pretrainer...")
        self.lsfa_pretrainer = LSFAPretrainer(
            backbone=self.backbone.backbone,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            temperature=config.temperature,
            adapt_layers=config.adapt_layers
        ).to(self.device)

        # ========== 加载数据 ==========
        print("\n📂 Loading unlabeled datasets...")

        # Rubber Wood数据
        rubber_dataset = UnlabeledWoodDataset(
            root_dir=config.rubber_wood_path,
            image_size=config.image_size
        )

        # Pine Wood数据
        pine_dataset = UnlabeledWoodDataset(
            root_dir=config.pine_wood_path,
            image_size=config.image_size
        )

        # 合并数据集
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([rubber_dataset, pine_dataset])

        self.train_loader = DataLoader(
            combined_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        print(f"✅ Total unlabeled samples: {len(combined_dataset)}")

        # ========== 优化器 ==========
        self.optimizer = optim.AdamW(
            self.lsfa_pretrainer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # ========== TensorBoard ==========
        self.writer = SummaryWriter(config.log_dir)

        self.best_loss = float('inf')

        print("\n✅ LSFA Trainer initialized")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.lsfa_pretrainer.train()

        total_loss = 0.0
        total_local_loss = 0.0
        total_global_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)

            # 前向传播
            outputs = self.lsfa_pretrainer(view1, view2)

            loss = outputs['total_loss']
            local_loss = outputs['local_loss']
            global_loss = outputs['global_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.lsfa_pretrainer.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_local_loss += local_loss.item()
            total_global_loss += global_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'local': f"{local_loss.item():.4f}",
                'global': f"{global_loss.item():.4f}"
            })

            # TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/TotalLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/LocalLoss', local_loss.item(), global_step)
                self.writer.add_scalar('Train/GlobalLoss', global_loss.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        avg_local_loss = total_local_loss / len(self.train_loader)
        avg_global_loss = total_global_loss / len(self.train_loader)

        return avg_loss, avg_local_loss, avg_global_loss

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'lsfa_state_dict': self.lsfa_pretrainer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }



        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'lsfa_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved best model: {best_path}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("STARTING LSFA PRETRAINING")
        print("=" * 70)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'=' * 70}")

            # 训练
            avg_loss, avg_local, avg_global = self.train_epoch(epoch)

            print(f"\n📊 Epoch {epoch} Results:")
            print(f"  - Total Loss: {avg_loss:.4f}")
            print(f"  - Local Loss: {avg_local:.4f}")
            print(f"  - Global Loss: {avg_global:.4f}")

            # 记录
            self.writer.add_scalar('Epoch/TotalLoss', avg_loss, epoch)
            self.writer.add_scalar('Epoch/LocalLoss', avg_local, epoch)
            self.writer.add_scalar('Epoch/GlobalLoss', avg_global, epoch)
            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=True)

            # 定期保存
            if epoch % 10 == 0:
                # self.save_checkpoint(epoch)
                print("...............")

            # 学习率衰减
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

        print("\n" + "=" * 70)
        print("🎉 LSFA PRETRAINING COMPLETED!")
        print(f"🏆 Best Loss: {self.best_loss:.4f}")
        print("=" * 70)

def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    from test_lsfa.lsfa_config import lsfa_config

    print("\n" + "=" * 70)
    print("LSFA SELF-SUPERVISED PRETRAINING")
    print("Based on ECCV 2024 Paper")
    print("=" * 70)

    print(f"\n📋 Configuration:")
    print(f"  - Epochs: {lsfa_config.num_epochs}")
    print(f"  - Batch size: {lsfa_config.batch_size}")
    print(f"  - Learning rate: {lsfa_config.learning_rate}")
    print(f"  - Temperature: {lsfa_config.temperature}")
    print(f"  - Adapt layers: {lsfa_config.adapt_layers}")

    trainer = LSFATrainer(lsfa_config)
    trainer.train()

if __name__ == "__main__":
    main()