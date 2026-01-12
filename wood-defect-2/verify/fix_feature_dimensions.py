#!/usr/bin/env python3
"""
自动修复特征维度不匹配问题
"""

import os
import sys
import re
import shutil
from datetime import datetime


# 首先检测实际的特征维度
def detect_actual_feature_dim():
    """检测DINOv2实际输出的特征维度"""
    print("=" * 80)
    print("步骤 1: 检测 DINOv2 实际特征维度")
    print("=" * 80)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import torch
    from models.backbone import VFMBackbone

    backbone = VFMBackbone(model_name='dinov2', freeze=True)
    test_image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = backbone(test_image, output_hidden_states=True)

    actual_dim = features[-1].shape[-1]
    num_layers = len(features)

    print(f"\n检测结果:")
    print(f"  实际特征维度: {actual_dim}")
    print(f"  层数: {num_layers}")

    # 判断是base还是large版本
    if actual_dim == 768 and num_layers == 13:  # 12层 + 1个embedding
        model_type = "DINOv2-base"
    elif actual_dim == 1024 and num_layers == 25:  # 24层 + 1个embedding
        model_type = "DINOv2-large"
    else:
        model_type = "Unknown"

    print(f"  模型类型: {model_type}")
    print("=" * 80 + "\n")

    return actual_dim, num_layers - 1, model_type  # -1因为要去掉embedding层


def backup_file(filepath):
    """备份文件"""
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        shutil.copy2(filepath, backup_path)
        print(f"  ✓ 已备份: {backup_path}")
        return backup_path
    return None


def fix_config_file(target_dim):
    """修复配置文件"""
    print("\n修复 configs/lam_config.py")
    print("-" * 80)

    filepath = "../configs/lam_config.py"
    backup_file(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复 feature_dim
    content = re.sub(
        r'feature_dim\s*=\s*\d+.*',
        f'feature_dim = {target_dim}  # Auto-fixed to match DINOv2 output',
        content
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ 已将 feature_dim 修改为 {target_dim}")


def fix_module_file(filepath, target_dim, param_name='feature_dim'):
    """修复模块文件中的默认feature_dim参数"""
    print(f"\n修复 {filepath}")
    print("-" * 80)

    if not os.path.exists(filepath):
        print(f"  ⚠️  文件不存在: {filepath}")
        return

    backup_file(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复默认参数值
    # 匹配 feature_dim=数字 的模式
    pattern = rf'{param_name}\s*=\s*\d+'
    replacement = f'{param_name}={target_dim}'

    new_content = re.sub(pattern, replacement, content)

    # 计算修改次数
    changes = len(re.findall(pattern, content))

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  ✓ 修改了 {changes} 处 {param_name} 默认值为 {target_dim}")


def fix_backbone_file(target_dim, num_layers):
    """修复backbone.py文件"""
    print(f"\n修复 models/backbone.py")
    print("-" * 80)

    filepath = "../models/backbone.py"
    backup_file(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 查找并修复self.feature_dim和self.num_layers
    for i, line in enumerate(lines):
        if 'self.feature_dim = ' in line and 'DINOv2' in lines[i - 1] if i > 0 else False:
            lines[i] = f'        self.feature_dim = {target_dim}  # Auto-fixed\n'
            print(f"  ✓ 第 {i + 1} 行: 已修改 feature_dim = {target_dim}")

        if 'self.num_layers = ' in line:
            lines[i] = f'        self.num_layers = {num_layers}  # Auto-fixed\n'
            print(f"  ✓ 第 {i + 1} 行: 已修改 num_layers = {num_layers}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    print("\n" + "=" * 80)
    print("LAM 特征维度自动修复工具")
    print("=" * 80 + "\n")

    # 检测实际维度
    actual_dim, num_layers, model_type = detect_actual_feature_dim()

    print(f"检测到 {model_type}，特征维度为 {actual_dim}")

    confirm = input(f"\n是否将所有配置和模块的 feature_dim 修改为 {actual_dim}? (y/n): ")

    if confirm.lower() != 'y':
        print("已取消修复")
        return

    print("\n" + "=" * 80)
    print("步骤 2: 开始修复文件")
    print("=" * 80)

    # 修复各个文件
    fix_config_file(actual_dim)
    fix_module_file('../models/fdm.py', actual_dim)
    fix_module_file('../models/iltm.py', actual_dim)
    fix_module_file('../models/lam.py', actual_dim)
    fix_module_file('../models/lsm.py', actual_dim)
    fix_backbone_file(actual_dim, num_layers)

    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)

    print(f"""
下一步操作建议:

1. 如果之前的模型是用错误的维度训练的，你需要重新训练:
   python train.py

2. 如果想用现有模型进行推理（会有维度转换）:
   python inference_with_dimension_fix.py \\
       --checkpoint /path/to/checkpoint \\
       --image /path/to/image.jpg

3. 测试修复后的配置:
   python -c "from configs.lam_config import config; print(f'feature_dim: {{config.feature_dim}}')"

备份文件说明:
- 所有被修改的文件都已自动备份
- 备份文件格式: 原文件名.backup_时间戳
- 如需恢复，可以手动复制备份文件
""")


if __name__ == '__main__':
    main()