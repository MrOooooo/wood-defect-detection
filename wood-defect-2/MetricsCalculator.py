import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

class MetricsCalculator:
    """指标计算工具类"""

    @staticmethod
    def calculate_metrics(pred: np.ndarray, label: np.ndarray, num_classes: int) -> Dict[str, float]:
        """
        计算分割指标
        Args:
            pred: 预测mask (H, W)
            label: 真实标签 (H, W)
            num_classes: 类别数
        Returns:
            metrics: 包含mIoU, mAcc, F1的字典
        """
        print("\n" + "=" * 60)
        print("mIoU 计算详细过程")
        print("=" * 60)

        # 忽略无效标签(如255)
        valid_mask = label < num_classes
        pred_valid = pred[valid_mask]
        label_valid = label[valid_mask]

        total_pixels = len(pred_valid)
        print(f"\n有效像素总数: {total_pixels:,}")
        print(f"图像尺寸: {pred.shape}")

        # 计算每个类别的IoU和Acc
        iou_list = []
        acc_list = []
        tp_total = 0
        fp_total = 0
        fn_total = 0

        print(f"\n{'类别':<12} {'真实像素':<12} {'预测像素':<12} {'交集':<12} {'并集':<12} {'IoU':<10} {'Acc':<10}")
        print("-" * 90)

        for class_id in range(num_classes):
            pred_mask = (pred_valid == class_id)
            label_mask = (label_valid == class_id)

            pred_count = np.sum(pred_mask)
            label_count = np.sum(label_mask)

            intersection = np.sum(pred_mask & label_mask)
            union = np.sum(pred_mask | label_mask)

            if union > 0:
                iou = intersection / union
                iou_list.append(iou)
                iou_str = f"{iou:.4f}"
            else:
                iou_str = "N/A"

            if label_count > 0:
                acc = intersection / label_count
                acc_list.append(acc)
                acc_str = f"{acc:.4f}"
            else:
                acc_str = "N/A"

            print(
                f"Class {class_id:<5} {label_count:<12,} {pred_count:<12,} {intersection:<12,} {union:<12,} {iou_str:<10} {acc_str:<10}")

            tp_total += intersection
            fp_total += np.sum(pred_mask & ~label_mask)
            fn_total += np.sum(~pred_mask & label_mask)

        print("-" * 90)

        # 计算指标
        mIoU = np.mean(iou_list) if iou_list else 0.0
        mAcc = np.mean(acc_list) if acc_list else 0.0

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n✓ 最终结果: mIoU={mIoU:.4f}, mAcc={mAcc:.4f}, F1={f1:.4f}")
        print("=" * 60 + "\n")

        return {
            'mIoU': mIoU,
            'mAcc': mAcc,
            'F1': f1,
            'precision': precision,
            'recall': recall
        }

    @staticmethod
    def calculate_class_distribution(pred_mask: np.ndarray, num_classes: int) -> List[float]:
        """计算类别分布"""
        total_pixels = pred_mask.size
        distribution = []

        for class_id in range(num_classes):
            count = np.sum(pred_mask == class_id)
            percentage = (count / total_pixels) * 100
            distribution.append(percentage)

        return distribution
