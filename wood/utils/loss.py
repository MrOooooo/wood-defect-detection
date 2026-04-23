"""
MODIFIED: segmentation losses with correct ignore_index handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, lambda_cov=0.5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cov = lambda_cov
        self.ignore_index = ignore_index

    def segmentation_loss(self, logits, labels):
        # MODIFIED: preserve ignore_index instead of clamping it into a valid class.
        valid_mask = labels != self.ignore_index
        if valid_mask.any():
            invalid_mask = valid_mask & ((labels < 0) | (labels >= self.num_classes))
            if invalid_mask.any():
                labels = labels.clone()
                labels[invalid_mask] = self.ignore_index

        ce_loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        dice_loss = self.dice_loss(logits, labels)
        return ce_loss + dice_loss

    def dice_loss(self, logits, labels):
        probs = F.softmax(logits, dim=1)

        valid_mask = labels != self.ignore_index
        safe_labels = labels.clone()
        safe_labels[~valid_mask] = 0

        labels_one_hot = F.one_hot(safe_labels, num_classes=self.num_classes)
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        labels_one_hot = labels_one_hot * valid_mask

        intersection = (probs * labels_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - dice.mean()

    def forward(self, logits, labels, cov_loss=None):
        seg_loss = self.segmentation_loss(logits, labels)
        total_loss = seg_loss

        loss_dict = {
            "seg_loss": seg_loss.item(),
            "total_loss": total_loss.item(),
        }

        if cov_loss is not None:
            total_loss = total_loss + self.lambda_cov * cov_loss
            loss_dict["cov_loss"] = cov_loss.item()
            loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        valid_mask = labels != self.ignore_index
        safe_labels = labels.clone()
        safe_labels[~valid_mask] = 0

        log_probs = F.log_softmax(logits, dim=1)
        log_probs = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma

        loss = -self.alpha * focal_weight * log_probs
        loss = loss * valid_mask.float()
        return loss.sum() / (valid_mask.sum() + 1e-5)
