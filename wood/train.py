import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.lam_config import config
from data.dataset import create_dataloader
from models.lam_mask2former import LAMMask2FormerModel
from utils.mask2former_loss import Mask2FormerLoss
from utils.metrics import SegmentationMetrics

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("CREATING PAPER-ORIGINAL MODEL")
        print("=" * 70)

        # MODIFIED: instantiate LSM modules once, then toggle them by stage.
        self.model = LAMMask2FormerModel(
            backbone_name=config.backbone,
            num_classes=config.num_classes,
            num_tokens=config.num_tokens,
            token_rank=config.token_rank,
            num_groups=config.num_groups,
            use_lsm=True,
            tau=config.tau,
            shared_tokens=True,
            adapt_layers=config.adapt_layers,
        )
        self._set_lsm_enabled(False)

        if config.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=config.gpu_ids)

        self.model = self.model.to(self.device)

        print("\n" + "=" * 70)
        print("LOADING DATASETS")
        print("=" * 70)

        self.train_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split="train",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            crop_range=config.crop_range,
            augmentation=True,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )

        self.val_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split="val",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            augmentation=False,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )

        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        # MODIFIED: use Mask2Former matching loss on the decoder outputs.
        self.criterion = Mask2FormerLoss(
            num_classes=config.num_classes,
            lambda_cov=config.lambda_cov,
            ignore_index=config.ignore_index,
        )

        trainable_params = self._model_ref().get_trainable_parameters()

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.eps,
        )
        self.scheduler = PolyLR(self.optimizer, max_epochs=config.num_epochs_pretrain, power=config.poly_power)

        self.metrics = SegmentationMetrics(num_classes=config.num_classes)
        self.writer = SummaryWriter(config.log_dir)

        self.best_miou = 0.0
        self.current_epoch = 0
        self.patience = 10
        self.patience_counter = 0

    def _model_ref(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _set_lsm_enabled(self, enabled):
        # MODIFIED: stage switching now toggles existing LSM modules instead of relying on missing modules.
        for lam in self._model_ref().multi_lam.lams:
            lam.use_lsm = enabled

    def _decode_semantic_logits(self, decoder_outputs, image_size):
        return self._model_ref().decode_semantic_logits(decoder_outputs, image_size)

    def _forward_decoder(self, images, compute_cov_loss):
        return self.model(
            images,
            compute_cov_loss=compute_cov_loss,
            return_decoder_outputs=True,
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_mask_loss = 0.0
        total_dice_loss = 0.0
        total_cov_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs, cov_loss = self._forward_decoder(images, compute_cov_loss=True)
            loss, loss_dict = self.criterion(outputs, labels, cov_loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += loss_dict["class_loss"]
            total_mask_loss += loss_dict["mask_loss"]
            total_dice_loss += loss_dict["dice_loss"]
            total_cov_loss += loss_dict.get("cov_loss", 0.0)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{loss_dict['class_loss']:.4f}",
                    "mask": f"{loss_dict['mask_loss']:.4f}",
                    "dice": f"{loss_dict['dice_loss']:.4f}",
                }
            )

            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar("Train/Loss", loss.item(), global_step)
                self.writer.add_scalar("Train/ClassLoss", loss_dict["class_loss"], global_step)
                self.writer.add_scalar("Train/MaskLoss", loss_dict["mask_loss"], global_step)
                self.writer.add_scalar("Train/DiceLoss", loss_dict["dice_loss"], global_step)
                if "cov_loss" in loss_dict:
                    self.writer.add_scalar("Train/CovLoss", loss_dict["cov_loss"], global_step)

        denom = len(self.train_loader)
        return {
            "loss": total_loss / denom,
            "class_loss": total_class_loss / denom,
            "mask_loss": total_mask_loss / denom,
            "dice_loss": total_dice_loss / denom,
            "cov_loss": total_cov_loss / denom,
        }

    def validate(self, epoch):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self._forward_decoder(images, compute_cov_loss=False)
                loss, _ = self.criterion(outputs, labels, cov_loss=None)
                total_loss += loss.item()

                semantic_logits = self._decode_semantic_logits(outputs, image_size=labels.shape[-2:])
                preds = torch.argmax(semantic_logits, dim=1)
                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        results = self.metrics.compute()
        avg_loss = total_loss / len(self.val_loader)

        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/mIoU", results["miou"], epoch)
        self.writer.add_scalar("Val/mAcc", results["macc"], epoch)
        self.writer.add_scalar("Val/F1", results["f1"], epoch)

        print("\nValidation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {results['miou']:.4f}")
        print(f"  mAcc: {results['macc']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  IoU per class: {results['iou_per_class']}")

        return results, avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        model_state = self._model_ref().state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou": self.best_miou,
            "configs": self.config,
        }

        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def _run_stage(self, start_epoch, end_epoch, stage_name):
        for epoch in range(start_epoch, end_epoch + 1):
            print(f"\n{'=' * 70}")
            print(f"{stage_name} - Epoch {epoch}")
            print(f"{'=' * 70}")

            train_stats = self.train_epoch(epoch)
            print(
                "Train Loss: "
                f"{train_stats['loss']:.4f} "
                f"(Cls: {train_stats['class_loss']:.4f}, "
                f"Mask: {train_stats['mask_loss']:.4f}, "
                f"Dice: {train_stats['dice_loss']:.4f}, "
                f"Cov: {train_stats['cov_loss']:.4f})"
            )

            if epoch % self.config.eval_freq == 0:
                val_results, _ = self.validate(epoch)

                if val_results["miou"] > self.best_miou:
                    self.best_miou = val_results["miou"]
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
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/LearningRate", current_lr, epoch)

    def train(self):
        print("\n" + "=" * 70)
        print("STAGE 1: PRE-TRAINING WITHOUT LSM")
        print("=" * 70)
        self._set_lsm_enabled(False)
        self._run_stage(1, self.config.num_epochs_pretrain, "Stage 1")

        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETED")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("STAGE 2: FULL TRAINING WITH LSM")
        print("=" * 70)

        self._set_lsm_enabled(True)

        # MODIFIED: continue from stage 1 optimizer state; only reset LR/scheduler for stage 2.
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate_stage2

        self.scheduler = PolyLR(
            self.optimizer,
            max_epochs=self.config.num_epochs_full,
            power=self.config.poly_power,
        )

        start_epoch = self.config.num_epochs_pretrain + 1
        end_epoch = self.config.num_epochs_pretrain + self.config.num_epochs_full
        self._run_stage(start_epoch, end_epoch, "Stage 2")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print("=" * 70)

        self.calculate_paper_table2_data()

    def calculate_paper_table2_data(self):
        print("\n" + "=" * 70)
        print("Calculating metrics for Paper Table 2")
        print("=" * 70)

        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Calculating metrics"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self._forward_decoder(images, compute_cov_loss=False)
                semantic_logits = self._decode_semantic_logits(outputs, image_size=labels.shape[-2:])
                preds = torch.argmax(semantic_logits, dim=1)
                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

        results = self.metrics.compute()

        print(f"mIoU: {results['miou']:.4f}")
        print(f"mAcc: {results['macc']:.4f}")
        print(f"F1: {results['f1']:.4f}")
        print(f"Overall Accuracy: {results['overall_acc']:.4f}")

        print("\nIoU per class:")
        for index, class_name in enumerate(self.config.rubber_classes):
            print(f"  {class_name}: {results['iou_per_class'][index]:.4f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("LAM + MASK2FORMER TRAINING (PAPER PATH)")
    print("=" * 70)

    config.update_for_dataset("rubber_wood")
    config.create_output_dirs()

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
