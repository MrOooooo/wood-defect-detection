# visualize.py
"""
论文可视化效果实现
包括：特征图、注意力图、t-SNE、性能对比等
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LAMSegmentationModel
from data.dataset import create_dataloader
from configs.lam_config import config


class Visualizer:
    """可视化工具类"""

    def __init__(self, checkpoint_path, dataset_name='pine_wood', device='cuda:1'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name

        # 加载模型
        print("Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        num_classes = 4 if dataset_name == 'pine_wood' else 6

        self.model = LAMSegmentationModel(
            backbone_name=config.backbone,
            num_classes=num_classes,
            num_tokens=config.num_tokens,
            token_rank=config.token_rank,
            num_groups=config.num_groups,
            use_lsm=True,
            tau=config.tau,
            shared_tokens=True
        )

        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 类别信息
        if dataset_name == 'pine_wood':
            self.class_names = ['Background', 'Dead Knot', 'Sound Knot', 'Missing Edge']
            self.class_colors = [
                [0, 0, 0],  # Background - 黑色
                [255, 0, 0],  # Dead Knot - 红色
                [0, 255, 0],  # Sound Knot - 绿色
                [0, 0, 255]  # Missing Edge - 蓝色
            ]
        else:
            self.class_names = ['Background', 'Dead Knot', 'Sound Knot',
                                'Missing Edge', 'Timber Core', 'Crack']
            self.class_colors = [
                [0, 0, 0],  # Background
                [255, 0, 0],  # Dead Knot
                [0, 255, 0],  # Sound Knot
                [0, 0, 255],  # Missing Edge
                [255, 255, 0],  # Timber Core
                [255, 0, 255]  # Crack
            ]

        print("Model loaded successfully!")

    def denormalize_image(self, image):
        """反归一化图像"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        if torch.is_tensor(image):
            image = image.cpu().numpy()

        if image.ndim == 4:  # Batch
            image = image[0]

        image = image.transpose(1, 2, 0)
        image = std * image + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image

    def label_to_color(self, label):
        """将标签转换为彩色图像"""
        if torch.is_tensor(label):
            label = label.cpu().numpy()

        if label.ndim == 3:  # Batch
            label = label[0]

        h, w = label.shape
        color_label = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in enumerate(self.class_colors):
            mask = label == class_id
            color_label[mask] = color

        return color_label

    def visualize_segmentation_results(self, dataloader, output_dir, num_samples=10):
        """
        可视化分割结果（论文图5、图6的风格）
        """
        print("\n=== Visualizing Segmentation Results ===")
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Generating results")):
                if idx >= num_samples:
                    break

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                filenames = batch['filename']

                # 前向传播
                logits = self.model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                # 转换为可视化格式
                img = self.denormalize_image(images[0])
                gt = self.label_to_color(labels[0])
                pred = self.label_to_color(preds[0])

                # 创建对比图
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # 原始图像
                axes[0].imshow(img)
                axes[0].set_title('(a) Input Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')

                # Ground Truth
                axes[1].imshow(gt)
                axes[1].set_title('(b) Ground Truth', fontsize=14, fontweight='bold')
                axes[1].axis('off')

                # 预测结果
                axes[2].imshow(pred)
                axes[2].set_title('(c) Segmentation Result', fontsize=14, fontweight='bold')
                axes[2].axis('off')

                # 叠加显示
                overlay = cv2.addWeighted(img, 0.6, pred, 0.4, 0)
                axes[3].imshow(overlay)
                axes[3].set_title('(d) Overlay', fontsize=14, fontweight='bold')
                axes[3].axis('off')

                plt.tight_layout()
                save_path = os.path.join(output_dir, f'result_{idx:03d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

        print(f"Results saved to {output_dir}")

    def extract_features_for_tsne(self, dataloader, num_samples=500):
        """
        提取特征用于t-SNE可视化（论文图1）
        """
        print("\n=== Extracting Features for t-SNE ===")

        features_list = []
        labels_list = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                if len(features_list) >= num_samples:
                    break

                images = batch['image'].to(self.device)

                # 提取backbone特征
                with torch.no_grad():
                    backbone_features = self.model.backbone(images, output_hidden_states=True)

                # 使用最后一层特征
                features = backbone_features[-1]  # (B, N, C)

                # 全局平均池化
                features = features.mean(dim=1)  # (B, C)

                features_list.append(features.cpu().numpy())

                # 标签：0=木材缺陷，1=自然场景（这里全是木材）
                labels_list.extend([0] * features.shape[0])

                if len(features_list) * features.shape[0] >= num_samples:
                    break

        features = np.concatenate(features_list, axis=0)[:num_samples]
        labels = np.array(labels_list)[:num_samples]

        return features, labels

    def visualize_tsne(self, features, labels, output_path, title="t-SNE Visualization"):
        """
        t-SNE可视化（论文图1）
        """
        print("\n=== Generating t-SNE Plot ===")

        # 如果特征维度太高，先用PCA降维
        if features.shape[1] > 50:
            print("Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)

        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # 绘图
        plt.figure(figsize=(10, 8))

        # 木材缺陷特征
        wood_mask = labels == 0
        plt.scatter(features_2d[wood_mask, 0], features_2d[wood_mask, 1],
                    c='#FFD700', alpha=0.6, s=50, label='Wood Defects',
                    edgecolors='black', linewidths=0.5)

        # 如果有自然场景数据
        if np.any(labels == 1):
            natural_mask = labels == 1
            plt.scatter(features_2d[natural_mask, 0], features_2d[natural_mask, 1],
                        c='#90EE90', alpha=0.6, s=50, label='Natural Scenes',
                        edgecolors='black', linewidths=0.5)

        plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"t-SNE plot saved to {output_path}")

    def visualize_feature_maps(self, image, output_dir):
        """
        可视化FDM和ILTM的特征图（论文图6、图7）
        """
        print("\n=== Visualizing Feature Maps ===")
        os.makedirs(output_dir, exist_ok=True)

        if not torch.is_tensor(image):
            image = torch.from_numpy(image).float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # 注册hook来获取中间特征
        features_before_fdm = []
        features_after_fdm = []
        features_after_iltm = []

        def hook_before_fdm(module, input, output):
            features_before_fdm.append(input[0].detach())

        def hook_after_fdm(module, input, output):
            if isinstance(output, tuple):
                features_after_fdm.append(output[0].detach())
            else:
                features_after_fdm.append(output.detach())

        def hook_after_iltm(module, input, output):
            features_after_iltm.append(output.detach())

        # 检查模型结构
        if not hasattr(self.model, 'multi_lam') or len(self.model.multi_lam.lams) == 0:
            print("Error: Model doesn't have LAM modules!")
            return

        # 注册hooks（以最后一层为例）
        last_lam = self.model.multi_lam.lams[-1]

        h1 = last_lam.fdm.register_forward_hook(hook_before_fdm)
        h2 = last_lam.fdm.register_forward_hook(hook_after_fdm)
        h3 = last_lam.iltm.register_forward_hook(hook_after_iltm)

        # 前向传播
        with torch.no_grad():
            _ = self.model(image, compute_cov_loss=False)

        # 移除hooks
        h1.remove()
        h2.remove()
        h3.remove()

        # 检查是否成功捕获特征
        if len(features_before_fdm) == 0:
            print("Error: No features captured before FDM!")
            print("This might be because LAM was not activated by LSM.")
            print("Try using a different image or disable LSM temporarily.")
            return

        if len(features_after_fdm) == 0:
            print("Error: No features captured after FDM!")
            return

        if len(features_after_iltm) == 0:
            print("Error: No features captured after ILTM!")
            return

        # 可视化特征图
        self._plot_feature_map(
            features_before_fdm[0][0],
            os.path.join(output_dir, 'feature_before_fdm.png'),
            'Feature Map Before FDM'
        )

        self._plot_feature_map(
            features_after_fdm[0][0],
            os.path.join(output_dir, 'feature_after_fdm.png'),
            'Feature Map After FDM'
        )

        self._plot_feature_map(
            features_after_iltm[0][0],
            os.path.join(output_dir, 'feature_after_iltm.png'),
            'Feature Map After ILTM'
        )

        print(f"Feature maps saved to {output_dir}")

    def _plot_feature_map(self, features, save_path, title, num_channels=16):
        """绘制特征图"""
        # features: (N, C)
        B, N, C = features.shape if features.ndim == 3 else (1, *features.shape)

        # 选择前num_channels个通道可视化
        H = W = int(np.sqrt(N))
        features = features.reshape(H, W, C)

        # 选择部分通道
        selected_channels = np.linspace(0, C - 1, num_channels, dtype=int)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        for idx, channel_idx in enumerate(selected_channels):
            feat = features[:, :, channel_idx].cpu().numpy()

            # 归一化到[0, 1]
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

            axes[idx].imshow(feat, cmap='jet')
            axes[idx].set_title(f'Channel {channel_idx}', fontsize=10)
            axes[idx].axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_similarity_maps(self, image, output_dir):
        """
        可视化ILTM的相似度图（论文图6 (d)(e)）
        """
        print("\n=== Visualizing Similarity Maps ===")
        os.makedirs(output_dir, exist_ok=True)

        if not torch.is_tensor(image):
            image = torch.from_numpy(image).float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # 修改ILTM来获取相似度图
        similarity_maps = []

        def hook_similarity(module, input, output):
            # 在ILTM forward中计算相似度图
            features = input[0]
            B, N, C = features.shape

            # 生成tokens
            tokens = torch.matmul(module.token_M, module.token_N)

            # 投影
            F_p = module.feature_proj(features)
            T_p = module.token_proj(tokens).unsqueeze(0).expand(B, -1, -1)

            # 计算相似度
            similarity = torch.matmul(F_p, T_p.transpose(1, 2))
            similarity = torch.softmax(similarity, dim=-1)

            similarity_maps.append(similarity.detach())

        # 注册hook
        last_lam = self.model.multi_lam.lams[-1]
        h = last_lam.iltm.register_forward_hook(hook_similarity)

        # 前向传播
        with torch.no_grad():
            _ = self.model(image, compute_cov_loss=False)

        h.remove()

        # 可视化相似度图
        if similarity_maps:
            sim_map = similarity_maps[0][0]  # (N, num_tokens)
            N, num_tokens = sim_map.shape
            H = W = int(np.sqrt(N))

            # 选择几个token可视化
            selected_tokens = [0, 10, 20, 30, 40, 50]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, token_idx in enumerate(selected_tokens):
                sim = sim_map[:, token_idx].reshape(H, W).cpu().numpy()

                axes[idx].imshow(sim, cmap='hot')
                axes[idx].set_title(f'Token {token_idx} Similarity', fontsize=12)
                axes[idx].axis('off')

                # 添加colorbar
                plt.colorbar(axes[idx].images[0], ax=axes[idx], fraction=0.046)

            plt.suptitle('ILTM Similarity Maps', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'similarity_maps.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Similarity maps saved to {output_dir}")

    def plot_performance_comparison(self, results_dict, output_path):
        """
        绘制性能对比图（论文表格和柱状图）
        """
        print("\n=== Plotting Performance Comparison ===")

        methods = list(results_dict.keys())
        metrics = ['mIoU', 'mACC', 'F1']

        # 准备数据
        data = {metric: [results_dict[m][metric] for m in methods]
                for metric in metrics}

        # 创建DataFrame
        df = pd.DataFrame(data, index=methods)

        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(methods))
        width = 0.25

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (idx - 1)
            bars = ax.bar(x + offset, df[metric], width,
                          label=metric, color=color, alpha=0.8)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Comparison on Pine Wood Dataset',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance comparison saved to {output_path}")

        # 同时保存表格
        table_path = output_path.replace('.png', '_table.csv')
        df.to_csv(table_path)
        print(f"Performance table saved to {table_path}")

    def plot_per_class_iou(self, iou_per_class, output_path):
        """
        绘制每个类别的IoU（论文表格中的详细结果）
        """
        print("\n=== Plotting Per-Class IoU ===")

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.class_names))
        bars = ax.bar(x, iou_per_class * 100, color='#3498db', alpha=0.8)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Defect Types', fontsize=12, fontweight='bold')
        ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class IoU Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Per-class IoU plot saved to {output_path}")

    def plot_confusion_matrix(self, confusion_matrix, output_path):
        """
        绘制混淆矩阵
        """
        print("\n=== Plotting Confusion Matrix ===")

        # 归一化混淆矩阵
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Normalized Count'},
                    ax=ax)

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize LAM Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='pine_wood',
                        choices=['pine_wood', 'rubber_wood'])
    parser.add_argument('--output_dir', type=str, default='/home/user4/桌面/wood-defect/wood-defect-output/visualizations')
    parser.add_argument('--vis_type', type=str, default='all',
                        choices=['all', 'segmentation', 'tsne', 'features',
                                 'similarity', 'comparison'])
    parser.add_argument('--num_samples', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建可视化器
    visualizer = Visualizer(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset
    )

    # 加载数据
    dataset_path = config.pine_wood_path if args.dataset == 'pine_wood' \
        else config.rubber_wood_path

    dataloader = create_dataloader(
        root_dir=dataset_path,
        split='val',
        batch_size=1,
        num_workers=0,
        image_size=config.image_size,
        augmentation=False,
        shuffle=False
    )

    # 执行可视化
    if args.vis_type in ['all', 'segmentation']:
        visualizer.visualize_segmentation_results(
            dataloader,
            os.path.join(args.output_dir, 'segmentation_results'),
            num_samples=args.num_samples
        )

    if args.vis_type in ['all', 'tsne']:
        features, labels = visualizer.extract_features_for_tsne(
            dataloader,
            num_samples=500
        )
        visualizer.visualize_tsne(
            features,
            labels,
            os.path.join(args.output_dir, 'tsne_plot.png')
        )

    if args.vis_type in ['all', 'features', 'similarity']:
        # 获取一张图像
        for batch in dataloader:
            sample_image = batch['image']
            break

        if args.vis_type in ['all', 'features']:
            visualizer.visualize_feature_maps(
                sample_image,
                os.path.join(args.output_dir, 'feature_maps')
            )

        if args.vis_type in ['all', 'similarity']:
            visualizer.visualize_similarity_maps(
                sample_image,
                os.path.join(args.output_dir, 'similarity_maps')
            )

    if args.vis_type in ['all', 'comparison']:
        # 示例性能数据（需要替换为实际结果）
        results_dict = {
            'U-Net': {'mIoU': 71.23, 'mACC': 84.56, 'F1': 82.45},
            'FCN': {'mIoU': 68.45, 'mACC': 82.34, 'F1': 80.12},
            'DeepLabv3+': {'mIoU': 78.91, 'mACC': 88.34, 'F1': 86.72},
            'Mask2Former': {'mIoU': 82.53, 'mACC': 89.67, 'F1': 87.53},
            'LAM (Ours)': {'mIoU': 85.32, 'mACC': 92.35, 'F1': 91.17}
        }

        visualizer.plot_performance_comparison(
            results_dict,
            os.path.join(args.output_dir, 'performance_comparison.png')
        )

    print(f"\nAll visualizations completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
python visualize/visualize.py \
    --checkpoint /home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth \
    --dataset rubber_wood \
    --output_dir /home/user4/桌面/wood-defect/wood-defect-output/visualizations \
    --vis_type segmentation \
    --num_samples 10 
"""