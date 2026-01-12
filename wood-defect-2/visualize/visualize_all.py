# visualize_all.py
"""
为所有测试图片生成可视化
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import WoodDefectInference
from data.dataset import create_dataloader
from configs.lam_config import config


def visualize_entire_dataset(checkpoint_path, dataset_type='pine', split='val'):
    """可视化整个数据集"""

    # 创建推理器
    print("Loading model...")
    inferencer = WoodDefectInference(
        checkpoint_path=checkpoint_path,
        dataset_type=dataset_type,
        device='cuda'
    )

    # 加载数据
    dataset_path = config.pine_wood_path if dataset_type == 'pine' \
        else config.rubber_wood_path

    dataloader = create_dataloader(
        root_dir=dataset_path,
        split=split,
        batch_size=1,
        num_workers=0,
        augmentation=False,
        shuffle=False
    )

    # 输出目录
    output_dir = f'./visualizations_all/{dataset_type}_{split}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nVisualizing {len(dataloader)} images...")
    print(f"Output: {output_dir}\n")

    # 处理每张图片
    for idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        images = batch['image'].to(inferencer.device)
        labels = batch['label'].to(inferencer.device)
        filename = batch['filename'][0]

        # 推理
        logits = inferencer.model(images, compute_cov_loss=False)
        preds = torch.argmax(logits, dim=1)

        # 转换为可视化格式
        img = inferencer.denormalize_image(images[0])
        gt = inferencer.label_to_color(labels[0])
        pred = inferencer.label_to_color(preds[0])

        # 保存可视化
        save_path = os.path.join(output_dir, f'{filename}_result.png')

        # 创建4联图
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(gt)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        overlay = cv2.addWeighted(img, 0.6, pred, 0.4, 0)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='pine',
                        choices=['pine', 'rubber'])
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])

    args = parser.parse_args()

    visualize_entire_dataset(
        checkpoint_path=args.checkpoint,
        dataset_type=args.dataset,
        split=args.split
    )