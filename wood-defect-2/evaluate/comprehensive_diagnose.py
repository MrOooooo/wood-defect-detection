# comprehensive_diagnose.py
"""
综合诊断脚本 - 定位evaluate_table2性能差的原因
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LAMSegmentationModel
from data.dataset import create_dataloader
from utils.metrics import SegmentationMetrics
from configs.lam_config import config


class PerformanceDiagnoser:
    def __init__(self, checkpoint_path, device='cuda:1'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path

        # 加载配置
        config.update_for_dataset('rubber_wood')
        self.config = config

        # 加载模型
        self.load_model()

        # 创建数据加载器
        self.val_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split='val',
            batch_size=1,
            num_workers=0,
            image_size=config.image_size,
            augmentation=False
        )

        self.metrics = SegmentationMetrics(num_classes=config.num_classes)

    def load_model(self):
        """加载模型并检查关键组件"""
        print("🔍 加载模型并检查架构...")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model = LAMSegmentationModel(
                backbone_name=config.backbone,
                num_classes=config.num_classes,
                num_tokens=config.num_tokens,
                token_rank=config.token_rank,
                num_groups=config.num_groups,
                use_lsm=True,
                tau=config.tau,
                shared_tokens=True
            )
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

        # 检查模型关键组件
        return self.check_model_components()

    def check_model_components(self):
        """检查模型关键组件是否正常"""
        issues = []

        # 检查LAM模块
        if not hasattr(self.model, 'multi_lam'):
            issues.append("❌ 缺少multi_lam模块")
        else:
            lam_modules = len(self.model.multi_lam.lams)
            print(f"✅ LAM模块数量: {lam_modules}")

            # 检查LSM是否启用
            for i, lam in enumerate(self.model.multi_lam.lams):
                if not lam.use_lsm:
                    issues.append(f"❌ 第{i}层LSM未启用")

        # 检查backbone
        if not hasattr(self.model, 'backbone'):
            issues.append("❌ 缺少backbone模块")
        else:
            print("✅ Backbone模块正常")

        if issues:
            print("\n".join(issues))
            return False
        return True

    def diagnose_data_issues(self):
        """诊断数据集问题"""
        print("\n🔍 诊断数据集问题...")

        label_stats = {
            'min': 999, 'max': -1,
            'class_counts': np.zeros(6),
            'problem_files': []
        }

        for batch in tqdm(self.val_loader, desc="扫描验证集"):
            labels = batch['label'].numpy()[0]
            filename = batch['filename'][0]

            batch_min, batch_max = labels.min(), labels.max()
            label_stats['min'] = min(label_stats['min'], batch_min)
            label_stats['max'] = max(label_stats['max'], batch_max)

            # 统计类别分布
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                if 0 <= u < 6:
                    label_stats['class_counts'][u] += c

            # 检查标签范围
            if batch_max >= 6 or batch_min < 0:
                label_stats['problem_files'].append({
                    'file': filename, 'min': batch_min, 'max': batch_max
                })

        # 输出诊断结果
        print(f"📊 标签范围: [{label_stats['min']}, {label_stats['max']}]")
        print(f"📊 类别分布: {label_stats['class_counts']}")

        if label_stats['problem_files']:
            print(f"❌ 发现{len(label_stats['problem_files'])}个问题文件")
        else:
            print("✅ 标签范围正常")

        return label_stats

    def diagnose_prediction_quality(self, num_samples=10):
        """诊断预测质量"""
        print(f"\n🔍 诊断预测质量（抽样{num_samples}张）...")

        self.metrics.reset()
        sample_results = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="评估预测")):
                if i >= num_samples:
                    break

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits = self.model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                # 更新指标
                self.metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

                # 分析单张图片
                pred_np = preds[0].cpu().numpy()
                label_np = labels[0].cpu().numpy()

                # 计算各类别IoU
                class_ious = []
                for class_id in range(6):
                    intersection = ((pred_np == class_id) & (label_np == class_id)).sum()
                    union = ((pred_np == class_id) | (label_np == class_id)).sum()
                    iou = intersection / (union + 1e-8)
                    class_ious.append(iou)

                sample_results.append({
                    'filename': batch['filename'][0],
                    'class_ious': class_ious,
                    'mean_iou': np.mean(class_ious)
                })

        # 分析结果
        overall_results = self.metrics.compute()
        print(f"📊 整体mIoU: {overall_results['miou']:.4f}")

        # 找出最差的样本
        worst_samples = sorted(sample_results, key=lambda x: x['mean_iou'])[:3]

        print("\n🔴 性能最差的3个样本:")
        for i, sample in enumerate(worst_samples):
            print(f"  {i + 1}. {sample['filename']}: mIoU={sample['mean_iou']:.4f}")
            for class_id, iou in enumerate(sample['class_ious']):
                if iou < 0.3:  # 识别率低的类别
                    print(f"    类别{class_id} IoU: {iou:.4f} ❌")

        return overall_results, sample_results

    def visualize_failure_cases(self, num_cases=5):
        """可视化失败案例"""
        print(f"\n🔍 可视化{num_cases}个失败案例...")

        os.makedirs('./diagnosis_output', exist_ok=True)
        failure_cases = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="寻找失败案例")):
                if len(failure_cases) >= num_cases:
                    break

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                # 计算单张mIoU
                pred_np = preds[0].cpu().numpy()
                label_np = labels[0].cpu().numpy()

                class_ious = []
                for class_id in range(6):
                    intersection = ((pred_np == class_id) & (label_np == class_id)).sum()
                    union = ((pred_np == class_id) | (label_np == class_id)).sum()
                    iou = intersection / (union + 1e-8)
                    class_ious.append(iou)

                mean_iou = np.mean(class_ious)

                # 记录失败案例（mIoU < 0.5）
                if mean_iou < 0.5:
                    failure_cases.append({
                        'image': images[0].cpu(),
                        'pred': preds[0].cpu(),
                        'label': labels[0].cpu(),
                        'filename': batch['filename'][0],
                        'miou': mean_iou
                    })

        # 可视化失败案例
        for i, case in enumerate(failure_cases):
            self.plot_comparison(case, f'./diagnosis_output/failure_case_{i}.png')

        print(f"✅ 保存了{len(failure_cases)}个失败案例可视化")
        return failure_cases

    def plot_comparison(self, case, save_path):
        """绘制预测对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        img = case['image'].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[0].imshow(img)
        axes[0].set_title(f"Original: {case['filename']}")
        axes[0].axis('off')

        # 预测结果
        axes[1].imshow(case['pred'].numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[1].set_title(f"Prediction (mIoU: {case['miou']:.3f})")
        axes[1].axis('off')

        # 真实标签
        axes[2].imshow(case['label'].numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def check_training_artifacts(self):
        """检查训练过程产物"""
        print("\n🔍 检查训练过程...")

        checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # 查找所有checkpoint
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
                epoch = int(f.split('_')[-1].split('.')[0])
                checkpoints.append((epoch, os.path.join(checkpoint_dir, f)))

        checkpoints.sort()

        if checkpoints:
            print(f"✅ 找到{len(checkpoints)}个checkpoint")
            print(f"   最早: epoch {checkpoints[0][0]}, 最晚: epoch {checkpoints[-1][0]}")

            # 检查最佳模型
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                print("✅ 找到best_model.pth")
                best_checkpoint = torch.load(best_model_path, map_location='cpu')
                if 'best_miou' in best_checkpoint:
                    print(f"✅ 最佳mIoU: {best_checkpoint['best_miou']:.4f}")
            else:
                print("❌ 未找到best_model.pth")
        else:
            print("❌ 未找到训练checkpoint")

    def run_comprehensive_diagnosis(self):
        """运行综合诊断"""
        print("=" * 80)
        print("🚀 开始综合性能诊断")
        print("=" * 80)

        # 1. 检查模型架构
        if not self.load_model():
            print("❌ 模型加载失败，停止诊断")
            return

        # 2. 诊断数据集
        data_stats = self.diagnose_data_issues()

        # 3. 诊断预测质量
        overall_results, sample_results = self.diagnose_prediction_quality()

        # 4. 可视化失败案例
        failure_cases = self.visualize_failure_cases()

        # 5. 检查训练过程
        self.check_training_artifacts()

        # 6. 生成诊断报告
        self.generate_diagnosis_report(data_stats, overall_results, failure_cases)

        print("=" * 80)
        print("✅ 诊断完成！查看 diagnosis_report.txt 获取详细建议")
        print("=" * 80)

    def generate_diagnosis_report(self, data_stats, overall_results, failure_cases):
        """生成诊断报告"""
        report = []

        report.append("=" * 80)
        report.append("📋 性能诊断报告")
        report.append("=" * 80)

        # 数据问题
        report.append("\n📊 数据诊断:")
        report.append(f"  标签范围: [{data_stats['min']}, {data_stats['max']}]")
        if data_stats['problem_files']:
            report.append(f"  ❌ 发现{len(data_stats['problem_files'])}个标签问题文件")
        else:
            report.append("  ✅ 标签范围正常")

        # 性能问题
        report.append(f"\n📊 性能诊断:")
        report.append(f"  整体mIoU: {overall_results['miou']:.4f}")
        report.append(f"  整体准确率: {overall_results['macc']:.4f}")

        if overall_results['miou'] < 0.7:
            report.append("  ❌ 性能严重低于论文水平(0.7668)")
        elif overall_results['miou'] < 0.76:
            report.append("  ⚠️ 性能略低于论文水平")
        else:
            report.append("  ✅ 性能达到论文水平")

        # 失败案例分析
        report.append(f"\n🔴 失败案例分析:")
        report.append(f"  发现{len(failure_cases)}个严重失败案例(mIoU < 0.5)")

        if failure_cases:
            for i, case in enumerate(failure_cases[:3]):
                report.append(f"  案例{i + 1}: {case['filename']} - mIoU: {case['miou']:.3f}")

        # 建议
        report.append("\n💡 修复建议:")

        if data_stats['problem_files']:
            report.append("  1. 修复标签问题文件（范围超出[0,5]）")

        if overall_results['miou'] < 0.7:
            report.append("  2. 重新训练模型，确保完整30轮训练")
            report.append("  3. 检查LSM模块是否在第二阶段启用")
            report.append("  4. 验证数据预处理与论文一致")

        if len(failure_cases) > 10:
            report.append("  5. 失败案例过多，建议检查数据集质量")

        report.append("\n" + "=" * 80)

        # 保存报告
        with open('./diagnosis_output/diagnosis_report.txt', 'w') as f:
            f.write('\n'.join(report))

        print('\n'.join(report))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='性能诊断工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='运行设备')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint不存在: {args.checkpoint}")
        return

    # 创建诊断器
    diagnoser = PerformanceDiagnoser(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 运行诊断
    diagnoser.run_comprehensive_diagnosis()


if __name__ == "__main__":
    main()

    """

    python evaluate/comprehensive_diagnose.py \
    --checkpoint  /home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth \
    --device cuda:1
    """