# inference_fixed.py
"""
使用训练好的模型对新图片进行推理 - 修复版
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LAMSegmentationModel
from configs.lam_config import config
import torchvision.transforms as T


class WoodDefectInference:
    """木材缺陷推理类"""

    def __init__(self, checkpoint_path, dataset_type='pine', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 确定类别数
        if dataset_type == 'pine':
            self.num_classes = 4
            self.class_names = ['background', 'dead_knot', 'sound_knot', 'missing_edge']
            self.class_colors = [
                [0, 0, 0],  # background
                [255, 0, 0],  # dead_knot
                [0, 255, 0],  # sound_knot
                [0, 0, 255]  # missing_edge
            ]
        else:  # rubber
            self.num_classes = 6
            self.class_names = ['background', 'dead_knot', 'sound_knot',
                                'missing_edge', 'timber_core', 'crack']
            self.class_colors = [
                [0, 0, 0],  # background
                [255, 0, 0],  # dead_knot
                [0, 255, 0],  # sound_knot
                [0, 0, 255],  # missing_edge
                [255, 255, 0],  # timber_core
                [255, 0, 255]  # crack
            ]

        # 从checkpoint获取配置（如果有）
        if 'configs' in checkpoint:
            saved_config = checkpoint['configs']
            adapt_layers = saved_config.adapt_layers if hasattr(saved_config, 'adapt_layers') else config.adapt_layers
        else:
            adapt_layers = config.adapt_layers

        # 创建模型
        print("Creating model...")
        self.model = LAMSegmentationModel(
            backbone_name=config.backbone,
            num_classes=self.num_classes,
            num_tokens=config.num_tokens,
            token_rank=config.token_rank,
            num_groups=config.num_groups,
            use_lsm=True,
            tau=config.tau,
            shared_tokens=True,
            adapt_layers=adapt_layers
        )

        # 智能加载权重
        try:
            # 首先尝试完全匹配
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✓ Loaded model with strict=True")
        except RuntimeError as e:
            print(f"Warning: Strict loading failed, trying flexible loading...")
            # 如果失败，使用灵活加载
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']

            # 过滤掉不匹配的键
            matched_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        matched_dict[k] = v
                    else:
                        print(f"  Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")
                else:
                    print(f"  Key not in model: {k}")

            # 更新模型字典
            model_dict.update(matched_dict)
            self.model.load_state_dict(model_dict)

            print(f"✓ Loaded {len(matched_dict)}/{len(model_dict)} parameters")

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded successfully!")
        print(f"✓ Training epoch: {checkpoint['epoch']}")
        if 'best_miou' in checkpoint:
            print(f"✓ Best mIoU: {checkpoint['best_miou']:.4f}")

        # 图像预处理
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """预处理图像"""
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size
        image_tensor = self.transform(original_image)
        return image_tensor, original_image, original_size

    def predict(self, image_path):
        """对单张图像进行预测"""
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor, compute_cov_loss=False)

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        confidence = probabilities.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Resize回原始大小
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )

        confidence = cv2.resize(
            confidence,
            original_size,
            interpolation=cv2.INTER_LINEAR
        )

        return prediction, confidence, original_image

    def visualize_result(self, original_image, prediction, confidence, save_path=None):
        """可视化预测结果"""
        original_np = np.array(original_image)
        h, w = prediction.shape
        colored_pred = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in enumerate(self.class_colors):
            mask = prediction == class_id
            colored_pred[mask] = color

        overlay = cv2.addWeighted(original_np, 0.6, colored_pred, 0.4, 0)
        max_confidence = confidence.max(axis=2)
        confidence_map = (max_confidence * 255).astype(np.uint8)
        confidence_colored = cv2.applyColorMap(confidence_map, cv2.COLORMAP_JET)
        confidence_colored = cv2.cvtColor(confidence_colored, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(original_np)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(colored_pred)
        axes[0, 1].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(confidence_colored)
        axes[1, 0].set_title('Confidence Map', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # 类别分布
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        class_percentages = []
        class_labels = []

        for class_id in range(self.num_classes):
            if class_id in unique:
                idx = np.where(unique == class_id)[0][0]
                percentage = counts[idx] / total_pixels * 100
            else:
                percentage = 0

            if percentage > 0.1:
                class_percentages.append(percentage)
                class_labels.append(self.class_names[class_id])

        axes[1, 1].bar(range(len(class_percentages)), class_percentages,
                       color=[np.array(self.class_colors[self.class_names.index(name)]) / 255
                              for name in class_labels])
        axes[1, 1].set_xticks(range(len(class_labels)))
        axes[1, 1].set_xticklabels(class_labels, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_title('Class Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        # 统计信息
        info_text = "Prediction Statistics:\n\n"
        for class_id in range(self.num_classes):
            if class_id in unique:
                idx = np.where(unique == class_id)[0][0]
                pixels = counts[idx]
                percentage = pixels / total_pixels * 100
                class_mask = prediction == class_id
                avg_conf = confidence[class_mask, class_id].mean()

                info_text += f"{self.class_names[class_id]}:\n"
                info_text += f"  Pixels: {pixels:,}\n"
                info_text += f"  Ratio: {percentage:.2f}%\n"
                info_text += f"  Confidence: {avg_conf:.3f}\n\n"

        axes[1, 2].text(0.1, 0.5, info_text, fontsize=10,
                        verticalalignment='center', fontfamily='monospace')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Result saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Wood Defect Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to image directory for batch inference')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user4/桌面/wood-defect/wood-defect-output/inference_results',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, default='pine',
                        choices=['pine', 'rubber'],
                        help='Dataset type')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image is None and args.image_dir is None:
        print("Error: Please specify either --image or --image_dir")
        return

    # 创建推理器
    inferencer = WoodDefectInference(
        checkpoint_path=args.checkpoint,
        dataset_type=args.dataset,
        device=args.device
    )

    # 单张图片推理
    if args.image:
        print(f"\n{'=' * 60}")
        print("Single Image Inference")
        print('=' * 60)

        prediction, confidence, original_image = inferencer.predict(args.image)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir,
                                   Path(args.image).stem + '_result.png')
        inferencer.visualize_result(original_image, prediction, confidence, output_path)

        print("\nPrediction Summary:")
        unique, counts = np.unique(prediction, return_counts=True)
        total = prediction.size

        for class_id, count in zip(unique, counts):
            percentage = count / total * 100
            class_name = inferencer.class_names[class_id]
            print(f"  {class_name:20s}: {count:8d} pixels ({percentage:6.2f}%)")

    print(f"\n{'=' * 60}")
    print("Inference Completed!")
    print('=' * 60)


if __name__ == "__main__":
    main()


"""
 python inference/inference.py \
    --checkpoint /home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth \
    --image "/home/user4/桌面/wood-defect/Picture/TestPicture/backImage/363.bmp" \
    --dataset rubber \
    --device cuda:1 
"""