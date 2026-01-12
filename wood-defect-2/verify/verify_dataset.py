# verify_dataset.py
"""
验证VOC格式数据集是否正确加载
并可视化一些样本
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wood_defect_segmentation.data.dataset import WoodDefectDataset, create_dataloader
from wood_defect_segmentation.configs.lam_config import config, auto_detect_classes


def visualize_sample(dataset, idx, save_path=None):
    """可视化单个样本"""
    sample = dataset[idx]

    image = sample['image']  # (3, H, W), normalized
    label = sample['label']  # (H, W)
    filename = sample['filename']

    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image.numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # 标签转为彩色
    label = label.numpy()
    num_classes = len(np.unique(label))

    # 生成颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    label_color = np.zeros((*label.shape, 3))

    for i, color in enumerate(colors):
        mask = label == i
        label_color[mask] = color[:3]

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title(f'Image: {filename}')
    axes[0].axis('off')

    axes[1].imshow(label, cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[1].set_title(f'Label (Classes: {np.unique(label)})')
    axes[1].axis('off')

    axes[2].imshow(label_color)
    axes[2].set_title('Label (Colored)')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_dataset(dataset, dataset_name):
    """分析数据集统计信息"""
    print(f"\n{'=' * 60}")
    print(f"Analyzing {dataset_name}")
    print(f"{'=' * 60}")

    print(f"Total samples: {len(dataset)}")

    # 统计标签分布
    all_labels = []
    image_sizes = []

    print("\nScanning dataset (this may take a while)...")
    for i in range(min(len(dataset), 100)):  # 只扫描前100张
        try:
            sample = dataset[i]
            label = sample['label'].numpy()
            all_labels.append(label.flatten())
            image_sizes.append(label.shape)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

    if not all_labels:
        print("No valid samples found!")
        return

    # 合并所有标签
    all_labels = np.concatenate(all_labels)

    # 统计类别分布
    unique_labels, counts = np.unique(all_labels, return_counts=True)

    print(f"\nLabel distribution (from {len(image_sizes)} samples):")
    total_pixels = counts.sum()

    for label, count in zip(unique_labels, counts):
        percentage = count / total_pixels * 100
        class_name = dataset.class_names[int(label)] if dataset.class_names and int(label) < len(
            dataset.class_names) else f"Class {label}"
        print(f"  {class_name:20s} (label={label}): {count:10d} pixels ({percentage:6.2f}%)")

    # 图像尺寸统计
    print(f"\nImage sizes (after transform):")
    unique_sizes = set(image_sizes)
    for size in unique_sizes:
        count = image_sizes.count(size)
        print(f"  {size}: {count} images")

    return unique_labels, counts


def verify_split_files(dataset_root):
    """验证split文件"""
    print(f"\nVerifying split files in {dataset_root}...")

    split_dir = os.path.join(dataset_root, 'ImageSets', 'Segmentation')

    if not os.path.exists(split_dir):
        print(f"Split directory not found: {split_dir}")
        return False

    splits = ['train', 'val', 'test']
    all_ids = set()

    for split in splits:
        split_file = os.path.join(split_dir, f'{split}.txt')

        if not os.path.exists(split_file):
            print(f" {split}.txt not found")
            continue

        with open(split_file, 'r') as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]

        print(f" {split}.txt: {len(ids)} images")

        # 检查重复
        duplicates = len(ids) - len(set(ids))
        if duplicates > 0:
            print(f"    ⚠ Warning: {duplicates} duplicate IDs found")

        all_ids.update(ids)

    print(f"\nTotal unique image IDs across all splits: {len(all_ids)}")
    return True


def check_file_existence(dataset_root, num_check=10):
    """检查图像和标签文件是否存在"""
    print(f"\nChecking file existence in {dataset_root}...")

    split_file = os.path.join(dataset_root, 'ImageSets', 'Segmentation', 'train.txt')

    if not os.path.exists(split_file):
        print(f"✗ train.txt not found")
        return False

    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

    image_dir = os.path.join(dataset_root, 'JPEGImages')
    label_dir = os.path.join(dataset_root, 'SegmentationClass')

    num_check = min(num_check, len(image_ids))
    print(f"Checking {num_check} samples...")

    missing_images = []
    missing_labels = []

    for image_id in image_ids[:num_check]:
        # 检查图像
        img_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(os.path.join(image_dir, image_id + ext)):
                img_found = True
                break

        if not img_found:
            missing_images.append(image_id)

        # 检查标签
        label_path = os.path.join(label_dir, image_id + '.png')
        if not os.path.exists(label_path):
            missing_labels.append(image_id)

    if missing_images:
        print(f" Missing images: {len(missing_images)}")
        for img_id in missing_images[:5]:
            print(f"    - {img_id}")
    else:
        print(f"All checked images found")

    if missing_labels:
        print(f" Missing labels: {len(missing_labels)}")
        for img_id in missing_labels[:5]:
            print(f"    - {img_id}")
    else:
        print(f"All checked labels found")

    return len(missing_images) == 0 and len(missing_labels) == 0


def main():
    parser = argparse.ArgumentParser(description='Verify Wood Defect Dataset')
    parser.add_argument('--dataset', type=str, default='pine',
                        choices=['pine', 'rubber', 'both'],
                        help='Which dataset to verify')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Which split to verify')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize some samples')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='E:\Dataset\wood-defect-output\dataset_verification',
                        help='Output directory for visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 选择数据集
    datasets_to_check = []
    if args.dataset == 'both':
        datasets_to_check = [
            ('pine', config.pine_wood_path),
            ('rubber', config.rubber_wood_path)
        ]
    elif args.dataset == 'pine':
        datasets_to_check = [('pine', config.pine_wood_path)]
    else:
        datasets_to_check = [('rubber', config.rubber_wood_path)]

    # 验证每个数据集
    for dataset_name, dataset_path in datasets_to_check:
        print("\n" + "=" * 70)
        print(f"VERIFYING {dataset_name.upper()} WOOD DATASET")
        print("=" * 70)

        # 检查路径
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            continue

        print(f"Dataset path exists: {dataset_path}")

        # 自动检测类别
        num_classes, class_names = auto_detect_classes(dataset_path)

        # 验证split文件
        verify_split_files(dataset_path)

        # 检查文件存在性
        check_file_existence(dataset_path, num_check=20)

        # 加载数据集
        try:
            print(f"\nLoading {args.split} dataset...")
            dataset = WoodDefectDataset(
                root_dir=dataset_path,
                split=args.split,
                image_size=512,
                augmentation=False  # 验证时不增强
            )

            print(f"✓ Dataset loaded successfully: {len(dataset)} samples")

            # 分析数据集
            analyze_dataset(dataset, f"{dataset_name} {args.split}")

            # 可视化样本
            if args.visualize and len(dataset) > 0:
                print(f"\nVisualizing {min(args.num_samples, len(dataset))} samples...")

                for i in range(min(args.num_samples, len(dataset))):
                    save_path = os.path.join(
                        args.output_dir,
                        f'{dataset_name}_{args.split}_sample_{i}.png'
                    )
                    try:
                        visualize_sample(dataset, i, save_path)
                    except Exception as e:
                        print(f"Error visualizing sample {i}: {e}")

                print(f"Visualizations saved to {args.output_dir}")

            # 测试DataLoader
            print("\nTesting DataLoader...")
            dataloader = create_dataloader(
                root_dir=dataset_path,
                split=args.split,
                batch_size=2,
                num_workers=0,
                augmentation=False,
                shuffle=False
            )

            # 加载一个batch
            for batch in dataloader:
                print(f"  DataLoader works!")
                print(f"  Batch shape: {batch['image'].shape}")
                print(f"  Label shape: {batch['label'].shape}")
                break

        except Exception as e:
            print(f" Error loading dataset: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()