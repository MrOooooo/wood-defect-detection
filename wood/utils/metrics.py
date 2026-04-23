# utils/metrics.py
"""
分割评估指标
"""

import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """分割评估指标计算类"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, labels):
        """
        更新混淆矩阵
        Args:
            preds: (B, H, W) 预测结果
            labels: (B, H, W) 真实标签
        """
        # 展平
        preds = preds.flatten()
        labels = labels.flatten()

        # 只计算有效区域（假设255是ignore label）
        mask = (labels >= 0) & (labels < self.num_classes)
        preds = preds[mask]
        labels = labels[mask]

        # 更新混淆矩阵
        cm = confusion_matrix(
            labels,
            preds,
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm

    def compute(self):
        """
        计算各项指标
        Returns:
            dict: 包含mIoU, mAcc, F1等指标
        """
        cm = self.confusion_matrix

        # IoU per class
        iou_per_class = np.diag(cm) / (
                cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-10
        )

        # Accuracy per class
        acc_per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-10)

        # Precision and Recall per class
        precision_per_class = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
        recall_per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-10)

        # F1 score per class
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (
                precision_per_class + recall_per_class + 1e-10
        )

        # Mean metrics
        miou = np.nanmean(iou_per_class)
        macc = np.nanmean(acc_per_class)
        f1 = np.nanmean(f1_per_class)

        # Overall accuracy
        overall_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)

        results = {
            'miou': miou,
            'macc': macc,
            'f1': f1,
            'overall_acc': overall_acc,
            'iou_per_class': iou_per_class,
            'acc_per_class': acc_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }

        return results

    def get_confusion_matrix(self):
        """返回混淆矩阵"""
        return self.confusion_matrix

