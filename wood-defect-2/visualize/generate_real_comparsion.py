# generate_real_comparison_figures.py
"""
使用真实基准测试结果生成论文对比图表
从benchmark_results中读取真实数据
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


class RealComparisonVisualizer:
    """使用真实数据的对比可视化器"""

    def __init__(self, benchmark_results_dir='./benchmark_results',
                 output_dir='./paper_figures_real'):
        self.benchmark_dir = benchmark_results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 13

        # 加载真实结果
        self.pine_results = self.load_results('pine_wood_results.json')
        self.rubber_results = self.load_results('rubber_wood_results.json')

    def load_results(self, filename):
        """加载基准测试结果"""
        filepath = os.path.join(self.benchmark_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using empty results.")
            return {}

        with open(filepath, 'r') as f:
            results = json.load(f)

        print(f"Loaded results from {filepath}")
        print(f"  Methods: {list(results.keys())}")

        return results

    def generate_table_1_pine_wood(self):
        """生成表1: Pine Wood数据集上的性能对比 - 使用真实数据"""
        print("\nGenerating Table 1 (Real Data): Pine Wood Results...")

        if not self.pine_results:
            print("Warning: No Pine Wood results available")
            return

        # 准备数据
        methods = list(self.pine_results.keys())
        data = {
            'Method': methods,
            'Dead Knot': [],
            'Sound Knot': [],
            'Missing Edge': [],
            'mIoU': [],
            'mACC': []
        }

        # 类别索引映射 (根据configs中的class_names)
        class_map = {
            'background': 0,
            'dead_knot': 1,
            'sound_knot': 2,
            'missing_edge': 3
        }

        for method in methods:
            results = self.pine_results[method]
            iou_per_class = results['iou_per_class']

            # 提取各类别IoU (跳过background)
            data['Dead Knot'].append(iou_per_class[class_map['dead_knot']] * 100)
            data['Sound Knot'].append(iou_per_class[class_map['sound_knot']] * 100)
            data['Missing Edge'].append(iou_per_class[class_map['missing_edge']] * 100)
            data['mIoU'].append(results['miou'] * 100)
            data['mACC'].append(results['macc'] * 100)

        df = pd.DataFrame(data)

        # 保存为CSV
        csv_path = os.path.join(self.output_dir, 'Table1_pine_wood_real.csv')
        df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"Saved CSV to {csv_path}")

        # 生成LaTeX表格
        latex_path = os.path.join(self.output_dir, 'Table1_pine_wood_real.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.2f'))
        print(f"Saved LaTeX to {latex_path}")

        # 可视化表格
        fig, ax = plt.subplots(figsize=(14, len(methods) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # 高亮最佳结果
        for i in range(len(df)):
            for j in range(1, len(df.columns)):
                cell = table[(i + 1, j)]
                # 如果是LAM (Ours)，高亮显示
                if 'LAM' in methods[i] or 'Ours' in methods[i]:
                    cell.set_facecolor('#90EE90')
                    cell.set_text_props(weight='bold')

        # 表头样式
        for j in range(len(df.columns)):
            cell = table[(0, j)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')

        plt.title('Table 1: Performance Comparison on Pine Wood Dataset (Real Data)',
                  fontsize=14, fontweight='bold', pad=20)

        plt.savefig(os.path.join(self.output_dir, 'Table1_pine_wood_real.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Table 1 (Real) saved")

        # 打印到控制台
        print("\n" + "=" * 80)
        print("TABLE 1: PINE WOOD DATASET RESULTS (REAL DATA)")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    def generate_table_2_rubber_wood(self):
        """生成表2: Rubber Wood数据集上的性能对比 - 使用真实数据"""
        print("\nGenerating Table 2 (Real Data): Rubber Wood Results...")

        if not self.rubber_results:
            print("Warning: No Rubber Wood results available")
            return

        methods = list(self.rubber_results.keys())
        data = {
            'Method': methods,
            'Dead Knot': [],
            'Sound Knot': [],
            'Missing Edge': [],
            'Timber Core': [],
            'Crack': [],
            'mIoU': [],
            'mACC': []
        }

        # 类别索引映射
        class_map = {
            'background': 0,
            'dead_knot': 1,
            'sound_knot': 2,
            'missing_edge': 3,
            'timber_core': 4,
            'crack': 5
        }

        for method in methods:
            results = self.rubber_results[method]
            iou_per_class = results['iou_per_class']

            data['Dead Knot'].append(iou_per_class[class_map['dead_knot']] * 100)
            data['Sound Knot'].append(iou_per_class[class_map['sound_knot']] * 100)
            data['Missing Edge'].append(iou_per_class[class_map['missing_edge']] * 100)
            data['Timber Core'].append(iou_per_class[class_map['timber_core']] * 100)
            data['Crack'].append(iou_per_class[class_map['crack']] * 100)
            data['mIoU'].append(results['miou'] * 100)
            data['mACC'].append(results['macc'] * 100)

        df = pd.DataFrame(data)

        # 保存
        csv_path = os.path.join(self.output_dir, 'Table2_rubber_wood_real.csv')
        df.to_csv(csv_path, index=False, float_format='%.2f')

        latex_path = os.path.join(self.output_dir, 'Table2_rubber_wood_real.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.2f'))

        print(f"Saved to {csv_path}")
        print("✓ Table 2 (Real) saved")

        # 打印到控制台
        print("\n" + "=" * 80)
        print("TABLE 2: RUBBER WOOD DATASET RESULTS (REAL DATA)")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    def generate_performance_bars(self):
        """生成性能对比柱状图 - 使用真实数据"""
        print("\nGenerating Performance Comparison Bars (Real Data)...")

        if not self.pine_results:
            print("Warning: No Pine Wood results available")
            return

        methods = list(self.pine_results.keys())
        miou = [self.pine_results[m]['miou'] * 100 for m in methods]
        macc = [self.pine_results[m]['macc'] * 100 for m in methods]
        f1 = [self.pine_results[m]['f1'] * 100 for m in methods]

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width, miou, width, label='mIoU',
                       color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, macc, width, label='mACC',
                       color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, f1, width, label='F1 Score',
                       color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)

        # 添加数值标签
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8, fontweight='bold')

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        ax.set_xlabel('Methods', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
        ax.set_title('Performance Comparison on Pine Wood Dataset (Real Data)',
                     fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend(loc='lower right', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])

        # 高亮最佳方法
        lam_idx = [i for i, m in enumerate(methods) if 'LAM' in m or 'Ours' in m]
        if lam_idx:
            ax.axvspan(lam_idx[0] - 0.5, lam_idx[0] + 0.5, alpha=0.1, color='gold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison_bars_real.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison_bars_real.pdf'),
                    bbox_inches='tight')
        plt.close()

        print("✓ Performance bars (Real) saved")

    def generate_per_class_comparison(self):
        """生成每个类别的IoU对比图"""
        print("\nGenerating Per-Class IoU Comparison (Real Data)...")

        if not self.pine_results:
            return

        methods = list(self.pine_results.keys())
        class_names = ['Dead Knot', 'Sound Knot', 'Missing Edge']

        # 准备数据 (跳过background)
        data = []
        for method in methods:
            iou_per_class = self.pine_results[method]['iou_per_class'][1:]  # 跳过background
            data.append(iou_per_class * 100)

        data = np.array(data)

        # 绘制分组柱状图
        x = np.arange(len(class_names))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for i, (method, color) in enumerate(zip(methods, colors)):
            offset = (i - len(methods) / 2) * width
            bars = ax.bar(x + offset, data[i], width, label=method,
                          color=color, alpha=0.85, edgecolor='black', linewidth=0.5)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Defect Types', fontweight='bold', fontsize=12)
        ax.set_ylabel('IoU (%)', fontweight='bold', fontsize=12)
        ax.set_title('Per-Class IoU Comparison on Pine Wood Dataset (Real Data)',
                     fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(loc='upper right', frameon=True, shadow=True, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'per_class_iou_comparison_real.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Per-class IoU comparison (Real) saved")

    def generate_all_figures(self):
        """生成所有对比图表"""
        print("\n" + "=" * 70)
        print("Generating All Comparison Figures with Real Data")
        print("=" * 70 + "\n")

        self.generate_table_1_pine_wood()
        self.generate_table_2_rubber_wood()
        self.generate_performance_bars()
        self.generate_per_class_comparison()

        print("\n" + "=" * 70)
        print("✅ All figures with real data generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate comparison figures using real benchmark results'
    )
    parser.add_argument('--benchmark_dir', type=str, default='./benchmark_results',
                        help='Directory containing benchmark results JSON files')
    parser.add_argument('--output_dir', type=str, default='./paper_figures_real',
                        help='Output directory for generated figures')
    args = parser.parse_args()

    # 检查基准测试结果是否存在
    if not os.path.exists(args.benchmark_dir):
        print(f"Error: Benchmark results directory not found: {args.benchmark_dir}")
        print("\nPlease run benchmark_comparison.py first:")
        print("  python benchmark_comparison.py --dataset pine_wood")
        print("  python benchmark_comparison.py --dataset rubber_wood")
        return

    # 创建可视化器并生成图表
    visualizer = RealComparisonVisualizer(
        benchmark_results_dir=args.benchmark_dir,
        output_dir=args.output_dir
    )

    visualizer.generate_all_figures()


if __name__ == "__main__":
    main()