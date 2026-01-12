"""
数据集诊断脚本 - 找出训练效果差的根本原因
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloader
from configs.lam_config import config


def diagnose_dataset():
    """全面诊断数据集问题"""

    print("=" * 80)
    print("DATASET DIAGNOSIS - Rubber Wood")
    print("=" * 80)

    config.update_for_dataset('rubber_wood')

    # 创建数据加载器
    train_loader = create_dataloader(
        root_dir=config.rubber_wood_path,
        split='train',
        batch_size=1,
        num_workers=0,
        image_size=config.image_size,
        augmentation=False
    )

    # 统计信息
    label_stats = {
        'min': 999,
        'max': -1,
        'class_counts': np.zeros(6),
        'problem_files': [],
        'empty_labels': [],
        'total_pixels': 0
    }

    print("\n1. Checking label ranges and class distribution...")

    for idx, batch in enumerate(tqdm(train_loader, desc="Scanning")):
        labels = batch['label'].numpy()[0]  # (H, W)
        filename = batch['filename'][0]

        # 检查范围
        batch_min = labels.min()
        batch_max = labels.max()

        label_stats['min'] = min(label_stats['min'], batch_min)
        label_stats['max'] = max(label_stats['max'], batch_max)
        label_stats['total_pixels'] += labels.size

        # 统计每个类别的像素数
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            if 0 <= u < 6:
                label_stats['class_counts'][u] += c

        # 检查问题
        if batch_max >= 6 or batch_min < 0:
            label_stats['problem_files'].append({
                'file': filename,
                'min': batch_min,
                'max': batch_max,
                'unique': unique.tolist()
            })

        # 检查是否所有标签都是背景
        if len(unique) == 1 and unique[0] == 0:
            label_stats['empty_labels'].append(filename)

    # 打印统计结果
    print("\n" + "=" * 80)
    print("DIAGNOSIS RESULTS")
    print("=" * 80)

    print(f"\n📊 Label Range: [{label_stats['min']}, {label_stats['max']}]")
    print(f"   Expected: [0, 5]")

    if label_stats['max'] >= 6 or label_stats['min'] < 0:
        print("   ❌ OUT OF RANGE! This will cause training issues!")
    else:
        print("   ✅ Range is valid")

    # 类别分布
    print(f"\n📊 Class Distribution:")
    class_names = ['Background', 'Dead Knot', 'Sound Knot',
                   'Missing Edge', 'Timber Core', 'Crack']

    total_defect_pixels = label_stats['class_counts'][1:].sum()

    for i, (name, count) in enumerate(zip(class_names, label_stats['class_counts'])):
        percentage = count / label_stats['total_pixels'] * 100
        print(f"   Class {i} ({name:15s}): {count:12,.0f} pixels ({percentage:5.2f}%)")

    print(f"\n   Total defect pixels: {total_defect_pixels:,.0f}")
    print(f"   Background ratio: {label_stats['class_counts'][0] / label_stats['total_pixels'] * 100:.2f}%")

    # 类别不平衡检查
    if label_stats['class_counts'][0] > 0.95 * label_stats['total_pixels']:
        print("\n   ⚠️ WARNING: Severe class imbalance! Background dominates!")

    # 检查是否有类别缺失
    missing_classes = [i for i, count in enumerate(label_stats['class_counts']) if count == 0]
    if missing_classes:
        print(f"\n   ⚠️ WARNING: Missing classes: {[class_names[i] for i in missing_classes]}")

    # 问题文件
    if label_stats['problem_files']:
        print(f"\n❌ Found {len(label_stats['problem_files'])} files with invalid labels:")
        for pf in label_stats['problem_files'][:5]:  # 只显示前5个
            print(f"   {pf['file']}: range [{pf['min']}, {pf['max']}], unique: {pf['unique']}")
        if len(label_stats['problem_files']) > 5:
            print(f"   ... and {len(label_stats['problem_files']) - 5} more")
    else:
        print("\n✅ No files with invalid labels")

    # 空标签文件
    if label_stats['empty_labels']:
        print(f"\n⚠️ Found {len(label_stats['empty_labels'])} files with only background:")
        for ef in label_stats['empty_labels'][:5]:
            print(f"   {ef}")
        if len(label_stats['empty_labels']) > 5:
            print(f"   ... and {len(label_stats['empty_labels']) - 5} more")

    return label_stats


def check_model_predictions():
    """检查模型预测情况"""

    print("\n" + "=" * 80)
    print("MODEL PREDICTION CHECK")
    print("=" * 80)

    checkpoint_path = '/home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth'

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    from models import LAMSegmentationModel

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = LAMSegmentationModel(
        backbone_name=config.backbone,
        num_classes=6,
        num_tokens=config.num_tokens,
        token_rank=config.token_rank,
        num_groups=config.num_groups,
        use_lsm=True,
        tau=config.tau,
        shared_tokens=True
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    # 测试几个样本
    val_loader = create_dataloader(
        root_dir=config.rubber_wood_path,
        split='val',
        batch_size=1,
        num_workers=0,
        image_size=config.image_size,
        augmentation=False
    )

    prediction_stats = {
        'predictions': np.zeros(6),
        'ground_truth': np.zeros(6)
    }

    print("\nChecking predictions on validation set...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Testing")):
            if idx >= 100:  # 只测试100个样本
                break

            images = batch['image'].to(device)
            labels = batch['label'].numpy()[0]

            logits = model(images, compute_cov_loss=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]

            # 统计
            for i in range(6):
                prediction_stats['predictions'][i] += (preds == i).sum()
                prediction_stats['ground_truth'][i] += (labels == i).sum()

    # 打印预测统计
    print("\n📊 Prediction vs Ground Truth:")
    class_names = ['Background', 'Dead Knot', 'Sound Knot',
                   'Missing Edge', 'Timber Core', 'Crack']

    for i, name in enumerate(class_names):
        pred_count = prediction_stats['predictions'][i]
        gt_count = prediction_stats['ground_truth'][i]

        if gt_count > 0:
            ratio = pred_count / gt_count
            status = "✅" if 0.5 < ratio < 2.0 else "⚠️"
        else:
            ratio = 0
            status = "❓"

        print(f"   {name:15s}: Pred={pred_count:10,.0f}, GT={gt_count:10,.0f}, "
              f"Ratio={ratio:.2f} {status}")

    # 检查是否模型输出都是背景
    total_pred = prediction_stats['predictions'].sum()
    bg_ratio = prediction_stats['predictions'][0] / total_pred

    if bg_ratio > 0.95:
        print(f"\n❌ CRITICAL: Model predicts {bg_ratio * 100:.1f}% background!")
        print("   This indicates the model is NOT learning defects properly!")
        print("\n   Possible causes:")
        print("   1. Learning rate too low or training not converged")
        print("   2. Loss function not working properly")
        print("   3. Class imbalance too severe")
        print("   4. Model architecture issues")


def visualize_samples():
    """可视化一些样本"""

    print("\n" + "=" * 80)
    print("SAMPLE VISUALIZATION")
    print("=" * 80)

    val_loader = create_dataloader(
        root_dir=config.rubber_wood_path,
        split='val',
        batch_size=1,
        num_workers=0,
        image_size=config.image_size,
        augmentation=False
    )

    class_colors = [
        [0, 0, 0],  # Background
        [255, 0, 0],  # Dead Knot
        [0, 255, 0],  # Sound Knot
        [0, 0, 255],  # Missing Edge
        [255, 255, 0],  # Timber Core
        [255, 0, 255]  # Crack
    ]

    print("\nGenerating sample visualizations...")

    # 找一些有缺陷的样本
    for idx, batch in enumerate(val_loader):
        labels = batch['label'].numpy()[0]

        # 跳过只有背景的样本
        if len(np.unique(labels)) == 1:
            continue

        # 可视化前3个有缺陷的样本
        if idx >= 3:
            break

        # 创建彩色标签图
        h, w = labels.shape
        colored_label = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in enumerate(class_colors):
            mask = labels == class_id
            colored_label[mask] = color

        # 保存
        output_dir = './diagnosis_output'
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        # 反归一化图像
        image = batch['image'][0].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(colored_label)
        plt.title('Ground Truth Label')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Saved sample visualizations to ./diagnosis_output/")


def main():
    """主诊断流程"""

    print("\n🔍 Starting comprehensive diagnosis...")
    print("This will help identify why your model performance is poor.\n")

    # 1. 诊断数据集
    label_stats = diagnose_dataset()

    # 2. 检查模型预测
    try:
        check_model_predictions()
    except Exception as e:
        print(f"\n⚠️ Could not check model predictions: {e}")

    # 3. 可视化样本
    try:
        visualize_samples()
    except Exception as e:
        print(f"\n⚠️ Could not visualize samples: {e}")

    # 总结建议
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # 基于诊断结果给出建议
    if label_stats['problem_files']:
        print("\n🔧 CRITICAL: Fix label files first!")
        print("   Run: Fix all files with labels >= 6")

    if label_stats['class_counts'][0] > 0.95 * label_stats['total_pixels']:
        print("\n🔧 Class imbalance is severe!")
        print("   Suggestions:")
        print("   1. Use focal loss instead of cross-entropy")
        print("   2. Increase lambda_cov to 1.0")
        print("   3. Use class weights in loss function")

    print("\n🔧 Training suggestions:")
    print("   1. Check if training loss is decreasing")
    print("   2. Verify learning rate schedule is working")
    print("   3. Make sure covariance loss is being calculated")
    print("   4. Try training longer (50+ epochs)")

    print("\n" + "=" * 80)
    print("Diagnosis complete! Check the output above for issues.")
    print("=" * 80)


if __name__ == "__main__":
    main()