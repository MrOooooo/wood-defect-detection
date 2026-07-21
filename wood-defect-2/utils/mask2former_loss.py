# utils/mask2former_loss.py
"""
Mask2Former专用损失函数
基于HuggingFace实现，适配木材缺陷分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class Mask2FormerLoss(nn.Module):
    """
    Mask2Former损失函数

    组成：
    1. 分类损失（Cross-Entropy）
    2. Mask损失（Binary CE）
    3. Dice损失
    4. 匈牙利匹配
    """

    def __init__(
            self,
            num_classes,
            lambda_cov=0.5,
            class_weight=2.0,
            mask_weight=5.0,
            dice_weight=5.0,
            num_points=12544,  # 用于计算mask loss的采样点数
            oversample_ratio=3.0,
            importance_sample_ratio=0.75
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_cov = lambda_cov

        # 损失权重
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight

        # Mask loss采样参数
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        print(f"\n🎯 Mask2Former Loss initialized:")
        print(f"  - Class weight: {class_weight}")
        print(f"  - Mask weight: {mask_weight}")
        print(f"  - Dice weight: {dice_weight}")
        print(f"  - Covariance weight: {lambda_cov}")

    def forward(self, outputs, targets, cov_loss=None):
        """
        计算总损失

        Args:
            outputs: Mask2Former输出，包含
                - class_queries_logits: (B, num_queries, num_classes+1)
                - masks_queries_logits: (B, num_queries, H, W)
            targets: (B, H, W) 语义标签
            cov_loss: LAM的协方差损失

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 提取预测
        pred_logits = outputs.class_queries_logits  # (B, Q, C+1)
        pred_masks = outputs.masks_queries_logits  # (B, Q, H, W)

        B, num_queries, num_classes_plus_one = pred_logits.shape
        _, _, H, W = pred_masks.shape

        # ========== 1. 将语义标签转换为实例格式 ==========
        target_instances = self._prepare_targets(targets)

        # ========== 2. 匈牙利匹配 ==========
        indices = self._hungarian_matching(
            pred_logits,
            pred_masks,
            target_instances
        )

        # ========== 3. 计算各项损失 ==========
        loss_ce = self._loss_labels(pred_logits, target_instances, indices)
        loss_mask = self._loss_masks(pred_masks, target_instances, indices)
        loss_dice = self._loss_dice(pred_masks, target_instances, indices)

        # ========== 4. 加权求和 ==========
        total_loss = (
                self.class_weight * loss_ce +
                self.mask_weight * loss_mask +
                self.dice_weight * loss_dice
        )

        # 添加协方差损失
        if cov_loss is not None:
            total_loss = total_loss + self.lambda_cov * cov_loss

        # 构造损失字典
        loss_dict = {
            'class_loss': loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            'mask_loss': loss_mask.item() if torch.is_tensor(loss_mask) else loss_mask,
            'dice_loss': loss_dice.item() if torch.is_tensor(loss_dice) else loss_dice,
            'total_loss': total_loss.item()
        }

        if cov_loss is not None:
            loss_dict['cov_loss'] = cov_loss.item() if torch.is_tensor(cov_loss) else cov_loss

        return total_loss, loss_dict

    def _prepare_targets(self, semantic_labels):
        """
        将语义标签转换为Mask2Former期望的实例格式

        Args:
            semantic_labels: (B, H, W) 语义标签

        Returns:
            targets: list of dict，每个dict包含：
                - labels: (num_instances,) 类别标签
                - masks: (num_instances, H, W) 二值mask
        """
        B, H, W = semantic_labels.shape
        targets = []

        for b in range(B):
            label_map = semantic_labels[b]  # (H, W)

            instance_labels = []
            instance_masks = []

            # 为每个类别创建一个pseudo-instance
            for class_id in range(1, self.num_classes):  # 跳过背景(0)
                mask = (label_map == class_id).float()

                if mask.sum() > 0:  # 该类别存在
                    instance_labels.append(class_id)
                    instance_masks.append(mask)

            # 转换为tensor
            if len(instance_labels) > 0:
                targets.append({
                    'labels': torch.tensor(instance_labels, device=semantic_labels.device, dtype=torch.long),
                    'masks': torch.stack(instance_masks)  # (num_instances, H, W)
                })
            else:
                # 没有前景实例，创建一个background instance
                targets.append({
                    'labels': torch.tensor([0], device=semantic_labels.device, dtype=torch.long),
                    'masks': torch.zeros(1, H, W, device=semantic_labels.device)
                })

        return targets

    @torch.no_grad()
    def _hungarian_matching(self, pred_logits, pred_masks, targets):
        """
        匈牙利匹配算法

        Args:
            pred_logits: (B, Q, C+1) 预测类别logits
            pred_masks: (B, Q, H, W) 预测masks
            targets: list of dict，目标实例

        Returns:
            indices: list of tuple (pred_idx, target_idx)
        """
        B, num_queries = pred_logits.shape[:2]

        indices = []

        for b in range(B):
            # 当前batch的预测
            out_prob = pred_logits[b].softmax(-1)  # (Q, C+1)
            out_mask = pred_masks[b]  # (Q, H, W)

            # 当前batch的目标
            tgt_ids = targets[b]['labels']  # (num_instances,)
            tgt_mask = targets[b]['masks']  # (num_instances, H, W)

            num_instances = len(tgt_ids)

            # ========== 计算cost矩阵 ==========
            # 1. 分类cost
            cost_class = -out_prob[:, tgt_ids]  # (Q, num_instances)

            # 2. Mask cost (使用Dice)
            out_mask_flat = out_mask.flatten(1)  # (Q, H*W)
            tgt_mask_flat = tgt_mask.flatten(1)  # (num_instances, H*W)

            # Dice cost
            numerator = 2 * torch.matmul(out_mask_flat, tgt_mask_flat.T)  # (Q, num_instances)
            denominator = out_mask_flat.sum(-1, keepdim=True) + tgt_mask_flat.sum(-1, keepdim=True).T
            cost_dice = 1 - numerator / (denominator + 1e-8)

            # 总cost
            cost = self.class_weight * cost_class + self.dice_weight * cost_dice
            cost = cost.cpu().numpy()

            # 匈牙利算法
            pred_idx, tgt_idx = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64, device=pred_logits.device),
                torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_logits.device)
            ))

        return indices

    def _loss_labels(self, pred_logits, targets, indices):
        """
        分类损失（Cross-Entropy）

        Args:
            pred_logits: (B, Q, C+1)
            targets: list of dict
            indices: list of tuple
        """
        # 收集所有匹配的predictions和targets
        src_logits = []
        target_classes = []

        B, num_queries, num_classes_plus_one = pred_logits.shape

        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_logits.append(pred_logits[b][src_idx])
            target_classes.append(targets[b]['labels'][tgt_idx])

        if len(src_logits) > 0:
            src_logits = torch.cat(src_logits)  # (total_matched, C+1)
            target_classes = torch.cat(target_classes)  # (total_matched,)

            loss = F.cross_entropy(src_logits, target_classes)
        else:
            loss = pred_logits.sum() * 0.0  # 返回0但保持梯度图

        return loss

    def _loss_masks(self, pred_masks, targets, indices):
        """
        Mask损失（Binary CE with point sampling）

        Args:
            pred_masks: (B, Q, H, W)
            targets: list of dict
            indices: list of tuple
        """
        src_masks = []
        target_masks = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_masks.append(pred_masks[b][src_idx])
            target_masks.append(targets[b]['masks'][tgt_idx])

        if len(src_masks) > 0:
            src_masks = torch.cat(src_masks)  # (total_matched, H, W)
            target_masks = torch.cat(target_masks)  # (total_matched, H, W)

            # Point sampling（可选，加速训练）
            # 这里简化，使用全部点
            loss = F.binary_cross_entropy_with_logits(
                src_masks,
                target_masks,
                reduction='mean'
            )
        else:
            loss = pred_masks.sum() * 0.0

        return loss

    def _loss_dice(self, pred_masks, targets, indices):
        """
        Dice损失

        Args:
            pred_masks: (B, Q, H, W)
            targets: list of dict
            indices: list of tuple
        """
        src_masks = []
        target_masks = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_masks.append(pred_masks[b][src_idx])
            target_masks.append(targets[b]['masks'][tgt_idx])

        if len(src_masks) > 0:
            src_masks = torch.cat(src_masks).sigmoid()  # (total_matched, H, W)
            target_masks = torch.cat(target_masks)  # (total_matched, H, W)

            # Flatten
            src_masks_flat = src_masks.flatten(1)  # (N, H*W)
            target_masks_flat = target_masks.flatten(1)  # (N, H*W)

            # Dice
            numerator = 2 * (src_masks_flat * target_masks_flat).sum(-1)
            denominator = src_masks_flat.sum(-1) + target_masks_flat.sum(-1)

            loss = 1 - (numerator / (denominator + 1e-8)).mean()
        else:
            loss = pred_masks.sum() * 0.0

        return loss


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Testing Mask2Former Loss...")

    # 模拟输出
    batch_size = 2
    num_queries = 100
    num_classes = 5
    H, W = 512, 512


    # 创建mock outputs
    class MockOutputs:
        def __init__(self):
            self.class_queries_logits = torch.randn(batch_size, num_queries, num_classes + 1)
            self.masks_queries_logits = torch.randn(batch_size, num_queries, H, W)


    outputs = MockOutputs()

    # 创建targets
    targets = torch.randint(0, num_classes, (batch_size, H, W))

    # 创建损失函数
    criterion = Mask2FormerLoss(
        num_classes=num_classes,
        lambda_cov=0.5
    )

    # 计算损失
    cov_loss = torch.tensor(0.1)
    total_loss, loss_dict = criterion(outputs, targets, cov_loss)

    print(f"\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n✅ Test passed!")