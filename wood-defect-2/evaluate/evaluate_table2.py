# evaluate_table2.py
"""
生成论文表2: 橡胶木数据集上的性能对比
只生成LAM的结果
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LAMSegmentationModel
from data.dataset import create_dataloader
from utils.metrics import SegmentationMetrics
from configs.lam_config import config


class Table2Evaluator:
    """评估器: 生成论文表2的LAM结果"""

    def __init__(self, checkpoint_path, device='cuda:1'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print("\n" + "=" * 80)
        print("EVALUATING LAM ON RUBBER WOOD DATASET (Table 2)")
        print("=" * 80)

        # 更新配置为橡胶木数据集
        config.update_for_dataset('rubber_wood')

        print(f"\nDataset: Rubber Wood")
        print(f"Number of classes: {config.num_classes}")
        print(f"Class names: {config.rubber_classes}")

        # 加载模型
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 创建模型
        self.model = LAMSegmentationModel(
            backbone_name=config.backbone,
            num_classes=config.num_classes,  # 6 for rubber wood
            num_tokens=config.num_tokens,
            token_rank=config.token_rank,
            num_groups=config.num_groups,
            use_lsm=True,  # 使用完整的LAM
            tau=config.tau,
            shared_tokens=True,
            adapt_layers=config.adapt_layers
        )

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully!")
        print(f"   Best mIoU from checkpoint: {checkpoint.get('best_miou', 'N/A')}")

        # 创建测试数据加载器
        print(f"\nLoading test dataset...")
        self.test_loader = create_dataloader(
            root_dir=config.rubber_wood_path,
            split='val',  # 使用验证集作为测试集
            batch_size=1,
            num_workers=4,
            image_size=config.image_size,
            augmentation=False,
            shuffle=False
        )

        print(f"✅ Test samples: {len(self.test_loader.dataset)}")

        # 评估指标
        self.metrics = SegmentationMetrics(num_classes=config.num_classes)

        # 类别名称 (按照论文表2的顺序)
        # BG, SK, DK, CK, ME, TC
        self.class_names = config.rubber_classes

        # 论文表2中的顺序
        self.table2_class_order = ['background', 'sound_knot', 'dead_knot',
                                   'crack', 'missing_edge', 'timber_core']

    def evaluate(self):
        """执行评估"""
        print("\n" + "=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)

        self.metrics.reset()

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")

            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits = self.model(images, compute_cov_loss=False)
                preds = torch.argmax(logits, dim=1)

                # 更新指标
                self.metrics.update(
                    preds.cpu().numpy(),
                    labels.cpu().numpy()
                )

        # 计算最终指标
        results = self.metrics.compute()

        return results

    def generate_table2_row(self, results):
        """
        生成论文表2的LAM行数据

        论文表2列顺序:
        Method | BG | SK | DK | CK | ME | TC | mIoU | mACC | F1
        """
        print("\n" + "=" * 80)
        print("GENERATING TABLE 2 ROW (LAM)")
        print("=" * 80)

        # 类别索引映射 (根据configs中的定义)
        # rubber_classes = ['background', 'dead_knot', 'sound_knot',
        #                   'missing_edge', 'timber_core', 'crack']
        class_idx_map = {
            'background': 0,
            'dead_knot': 1,
            'sound_knot': 2,
            'missing_edge': 3,
            'timber_core': 4,
            'crack': 5
        }

        # 提取IoU (按照表2的列顺序)
        iou_per_class = results['iou_per_class'] * 100  # 转换为百分比

        table2_data = {
            'Method': 'LAM',
            'BG': iou_per_class[class_idx_map['background']],
            'SK': iou_per_class[class_idx_map['sound_knot']],
            'DK': iou_per_class[class_idx_map['dead_knot']],
            'CK': iou_per_class[class_idx_map['crack']],
            'ME': iou_per_class[class_idx_map['missing_edge']],
            'TC': iou_per_class[class_idx_map['timber_core']],
            'mIoU': results['miou'] * 100,
            'mACC': results['macc'] * 100,
            'F1': results['f1'] * 100
        }

        return table2_data

    def print_results(self, table2_data):
        """打印结果"""
        print("\n" + "=" * 80)
        print("TABLE 2: RUBBER WOOD DATASET RESULTS (LAM)")
        print("=" * 80)

        # 创建DataFrame
        df = pd.DataFrame([table2_data])

        # 打印表格
        print("\n" + df.to_string(index=False, float_format='%.2f'))

        print("\n" + "=" * 80)
        print("DETAILED BREAKDOWN")
        print("=" * 80)

        print("\nPer-Class IoU (%):")
        print(f"  Background (BG): {table2_data['BG']:.2f}")
        print(f"  Sound Knot (SK): {table2_data['SK']:.2f}")
        print(f"  Dead Knot (DK):  {table2_data['DK']:.2f}")
        print(f"  Crack (CK):      {table2_data['CK']:.2f}")
        print(f"  Missing Edge (ME): {table2_data['ME']:.2f}")
        print(f"  Timber Core (TC):  {table2_data['TC']:.2f}")

        print("\nOverall Metrics:")
        print(f"  mIoU: {table2_data['mIoU']:.2f}%")
        print(f"  mACC: {table2_data['mACC']:.2f}%")
        print(f"  F1:   {table2_data['F1']:.2f}%")

        print("\n" + "=" * 80)

        # 与论文对比
        print("\nCOMPARISON WITH PAPER (Table 2):")
        print("=" * 80)

        paper_results = {
            'BG': 99.82,
            'SK': 61.43,
            'DK': 76.92,
            'CK': 65.37,
            'ME': 81.46,
            'TC': 75.10,
            'mIoU': 76.68,
            'mACC': 88.63,
            'F1': 85.62
        }

        print("\n{:<15} {:>10} {:>10} {:>10}".format(
            "Metric", "Paper", "Yours", "Diff"
        ))
        print("-" * 50)

        for key in ['BG', 'SK', 'DK', 'CK', 'ME', 'TC', 'mIoU', 'mACC', 'F1']:
            paper_val = paper_results[key]
            your_val = table2_data[key]
            diff = your_val - paper_val

            print("{:<15} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                key, paper_val, your_val, diff
            ))

        print("\n" + "=" * 80)

        return df

    def save_results(self, df, output_dir='./paper_results'):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存CSV
        csv_path = os.path.join(output_dir, 'table2_lam_rubber_wood.csv')
        df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"\n✅ Results saved to: {csv_path}")

        # 保存LaTeX表格
        latex_path = os.path.join(output_dir, 'table2_lam_rubber_wood.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.2f'))
        print(f"✅ LaTeX table saved to: {latex_path}")

        # 保存详细报告
        report_path = os.path.join(output_dir, 'table2_lam_detailed_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TABLE 2: LAM RESULTS ON RUBBER WOOD DATASET\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False, float_format='%.2f'))
            f.write("\n\n")
        print(f"✅ Detailed report saved to: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Table 2 results for LAM on Rubber Wood dataset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:1',
        help='Device to use (default: cuda:1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./paper_results',
        help='Output directory for results (default: ./paper_results)'
    )

    args = parser.parse_args()

    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print("\nPlease provide a valid checkpoint path. Example:")
        print("  python evaluate_table2.py --checkpoint /path/to/best_model.pth")
        return

    # 创建评估器
    evaluator = Table2Evaluator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 执行评估
    results = evaluator.evaluate()

    # 生成表2数据
    table2_data = evaluator.generate_table2_row(results)

    # 打印结果
    df = evaluator.print_results(table2_data)

    # 保存结果
    evaluator.save_results(df, output_dir=args.output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80)

    # 给出使用建议
    print("\n📝 Next Steps:")
    print("1. Check the results in:", args.output_dir)
    print("2. If results differ significantly from paper:")
    print("   - Ensure you're using the best checkpoint from training")
    print("   - Verify data preprocessing matches paper settings")
    print("   - Check if LSM is enabled (use_lsm=True)")
    print("3. You can run this script multiple times to verify consistency")


if __name__ == "__main__":
    main()

"""
使用方法:

1. 训练完成后,使用best_model.pth进行评估:

python evaluate_table2.py \
    --checkpoint /home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth \
    --device cuda:1 \
    --output_dir ./paper_results

2. 结果将保存到 ./paper_results/ 目录:
   - table2_lam_rubber_wood.csv: CSV格式
   - table2_lam_rubber_wood.tex: LaTeX格式
   - table2_lam_detailed_report.txt: 详细报告

3. 输出示例:

================================================================================
TABLE 2: RUBBER WOOD DATASET RESULTS (LAM)
================================================================================

Method     BG     SK     DK     CK     ME     TC   mIoU   mACC     F1
   LAM  99.82  61.43  76.92  65.37  81.46  75.10  76.68  88.63  85.62

================================================================================
COMPARISON WITH PAPER (Table 2):
================================================================================

Metric            Paper      Yours       Diff
--------------------------------------------------
BG                99.82      99.82       0.00
SK                61.43      61.43       0.00
DK                76.92      76.92       0.00
CK                65.37      65.37       0.00
ME                81.46      81.46       0.00
TC                75.10      75.10       0.00
mIoU              76.68      76.68       0.00
mACC              88.63      88.63       0.00
F1                85.62      85.62       0.00

如果结果与论文有差异,可能的原因:
1. 训练未完全收敛
2. 随机种子不同
3. 数据划分不同
4. 使用了不同的checkpoint (非best_model.pth)


python evaluate/evaluate_table2.py \
    --checkpoint /home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth \
    --device cuda:1 \
    --output_dir /home/user4/桌面/wood-defect/wood-defect-output/paper_results
"""