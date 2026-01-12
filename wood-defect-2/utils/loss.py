
# utils/loss.py
"""
分割损失函数
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """
    分割损失函数
    包含交叉熵损失和协方差正则化损失
    """

    def __init__(self, num_classes, lambda_cov=0.5, ignore_index=255):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_cov = lambda_cov
        self.ignore_index = ignore_index

        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def segmentation_loss(self, logits, labels):
        """
        计算分割损失
        Args:
            logits: (B, C, H, W) 预测logits
            labels: (B, H, W) 真实标签
        Returns:
            loss: 标量损失
        """
        # ⭐ 添加标签范围检查
        if labels.max() >= self.num_classes:
            print(f"Warning: Label max {labels.max()} >= num_classes {self.num_classes}")
            labels = torch.clamp(labels, 0, self.num_classes - 1)

        if labels.min() < 0:
            print(f"Warning: Label min {labels.min()} < 0")
            labels = torch.clamp(labels, 0, self.num_classes - 1)

        # 交叉熵损失
        ce_loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        # Dice损失
        dice_loss = self.dice_loss(logits, labels)

        return ce_loss + dice_loss
        # return self.ce_loss(logits, labels)

    def dice_loss(self, logits, labels):
        """
        Dice损失（可选）
        Args:
            logits: (B, C, H, W) 预测logits
            labels: (B, H, W) 真实标签
        Returns:
            loss: 标量损失
        """
        # Softmax
        probs = F.softmax(logits, dim=1)

        # One-hot编码标签
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes)
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

        # 计算Dice系数
        intersection = (probs * labels_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

        # Dice loss
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def forward(self, logits, labels, cov_loss=None):
        """
        完整损失计算
        Args:
            logits: (B, C, H, W) 预测logits
            labels: (B, H, W) 真实标签
            cov_loss: 协方差损失（可选）
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 分割损失
        seg_loss = self.segmentation_loss(logits, labels)

        # 总损失
        total_loss = seg_loss

        loss_dict = {
            'seg_loss': seg_loss.item(),
            'total_loss': total_loss.item()
        }

        # 添加协方差损失
        if cov_loss is not None:
            total_loss = total_loss + self.lambda_cov * cov_loss
            loss_dict['cov_loss'] = cov_loss.item()
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W)
        """
        # 计算log softmax
        log_probs = F.log_softmax(logits, dim=1)

        # 获取对应类别的log概率
        log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # 计算概率
        probs = torch.exp(log_probs)

        # Focal权重
        focal_weight = (1 - probs) ** self.gamma

        # 损失
        loss = -self.alpha * focal_weight * log_probs

        # 忽略特定索引
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum() + 1e-5)
        else:
            loss = loss.mean()

        return loss


if __name__ == "__main__":
    # 测试指标
    print("Testing Segmentation Metrics...")

    metrics = SegmentationMetrics(num_classes=5)

    # 创建测试数据
    batch_size = 2
    height = 512
    width = 512

    # 模拟预测和标签
    preds = np.random.randint(0, 5, size=(batch_size, height, width))
    labels = np.random.randint(0, 5, size=(batch_size, height, width))

    # 更新指标
    metrics.update(preds, labels)

    # 计算结果
    results = metrics.compute()

    print(f"mIoU: {results['miou']:.4f}")
    print(f"mAcc: {results['macc']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"Overall Acc: {results['overall_acc']:.4f}")
    print(f"IoU per class: {results['iou_per_class']}")

    print("\nTesting Segmentation Loss...")

    # 创建损失函数
    criterion = SegmentationLoss(num_classes=5, lambda_cov=0.5)

    # 创建测试数据
    logits = torch.randn(batch_size, 5, height, width)
    labels = torch.randint(0, 5, (batch_size, height, width))
    cov_loss = torch.tensor(0.1)

    # 计算损失
    total_loss, loss_dict = criterion(logits, labels, cov_loss)

    print(f"Segmentation Loss: {loss_dict['seg_loss']:.4f}")
    print(f"Covariance Loss: {loss_dict['cov_loss']:.4f}")
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")

    print("\nAll tests passed!")