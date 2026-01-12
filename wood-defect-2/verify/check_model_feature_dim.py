#!/usr/bin/env python3
"""
检查模型特征维度和配置
找出预测全黑的根本原因
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.backbone import VFMBackbone
from models import LAMSegmentationModel
from configs.lam_config import config

print("=" * 80)
print("检查 DINOv2 Backbone 特征维度")
print("=" * 80)

# 1. 检查 Backbone
print("\n1. 创建 Backbone 并检查特征...")
backbone = VFMBackbone(model_name='dinov2', freeze=True, output_layers=None)

# 测试图像
test_image = torch.randn(1, 3, 512, 512)
print(f"测试图像形状: {test_image.shape}")

# 前向传播
with torch.no_grad():
    features = backbone(test_image, output_hidden_states=True)

print(f"\n特征层数: {len(features)}")
print(f"第一层特征形状: {features[0].shape}")
print(f"最后一层特征形状: {features[-1].shape}")

# 检查特征维度
B, N, C = features[-1].shape
print(f"\n✓ 实际特征维度: {C}")
print(f"✓ 配置中的特征维度: {config.feature_dim}")

if C != config.feature_dim:
    print(f"\n⚠️  警告：特征维度不匹配！")
    print(f"   实际: {C}, 配置: {config.feature_dim}")
    print(f"   这会导致模型无法正常工作！")
else:
    print(f"\n✓ 特征维度匹配")

# 2. 检查已训练的模型
print("\n" + "=" * 80)
print("2. 检查已训练模型的配置")
print("=" * 80)

checkpoint_path = '/home/user4/桌面/wood-defect/wood-defect-output/checkpoints/best_model.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\n模型训练轮数: {checkpoint['epoch']}")
    if 'best_miou' in checkpoint:
        print(f"最佳 mIoU: {checkpoint['best_miou']:.4f}")

    if 'configs' in checkpoint:
        saved_config = checkpoint['configs']
        print(f"\n保存的配置:")
        print(f"  num_classes: {saved_config.num_classes}")
        print(f"  num_tokens: {saved_config.num_tokens}")
        print(f"  token_rank: {saved_config.token_rank}")
        print(f"  num_groups: {saved_config.num_groups}")
        print(f"  feature_dim: {saved_config.feature_dim}")
        print(f"  adapt_layers: {saved_config.adapt_layers}")

        if saved_config.feature_dim != C:
            print(f"\n⚠️  严重错误：模型训练时使用的特征维度 ({saved_config.feature_dim}) "
                  f"与实际backbone输出 ({C}) 不匹配！")
            print(f"   这会导致模型无法正确预测！")

    # 检查模型权重的形状
    print(f"\n模型权重键（前20个）:")
    for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:20]):
        shape = checkpoint['model_state_dict'][key].shape
        print(f"  {i + 1}. {key}: {shape}")

    # 检查 LAM 相关的权重
    print(f"\nLAM 模块权重:")
    lam_keys = [k for k in checkpoint['model_state_dict'].keys() if 'multi_lam' in k]
    for key in lam_keys[:10]:
        shape = checkpoint['model_state_dict'][key].shape
        print(f"  {key}: {shape}")
else:
    print(f"\n⚠️  找不到checkpoint文件: {checkpoint_path}")

# 3. 尝试创建并加载模型
print("\n" + "=" * 80)
print("3. 尝试创建模型并加载权重")
print("=" * 80)

try:
    # 使用当前配置创建模型
    print(f"\n使用配置: feature_dim={config.feature_dim}")
    model = LAMSegmentationModel(
        backbone_name='dinov2',
        num_classes=4,
        num_tokens=config.num_tokens,
        token_rank=config.token_rank,
        num_groups=config.num_groups,
        use_lsm=True,
        tau=config.tau,
        shared_tokens=True,
        adapt_layers=config.adapt_layers
    )

    print("✓ 模型创建成功")

    # 尝试加载权重
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✓ 模型权重加载成功 (strict=True)")
        except Exception as e:
            print(f"✗ 严格加载失败: {str(e)[:200]}")

            # 尝试宽松加载
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=False
            )
            print(f"\n宽松加载结果:")
            print(f"  缺失的键: {len(missing_keys)}")
            print(f"  意外的键: {len(unexpected_keys)}")

            if missing_keys:
                print(f"\n  缺失键示例（前5个）:")
                for key in missing_keys[:5]:
                    print(f"    - {key}")

            if unexpected_keys:
                print(f"\n  意外键示例（前5个）:")
                for key in unexpected_keys[:5]:
                    print(f"    - {key}")

    # 测试前向传播
    print(f"\n测试前向传播...")
    model.eval()
    with torch.no_grad():
        test_output = model(test_image, compute_cov_loss=False)

    print(f"✓ 前向传播成功")
    print(f"  输出形状: {test_output.shape}")

    # 检查输出
    pred = torch.argmax(test_output, dim=1)
    unique_classes = torch.unique(pred)
    print(f"  预测的类别: {unique_classes.tolist()}")

    if len(unique_classes) == 1 and unique_classes[0] == 0:
        print(f"\n⚠️  警告：模型只预测背景类！")
        print(f"  这说明模型没有正确学习或配置有问题")

except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("诊断建议")
print("=" * 80)

print("""
如果发现特征维度不匹配：
1. 修改 configs/lam_config.py 中的 feature_dim 为正确值
2. 重新训练模型（或使用正确的feature_dim加载）

如果模型只预测背景：
1. 检查训练日志，确认模型是否正确训练
2. 在推理时禁用 LSM
3. 检查数据预处理是否一致

如果权重加载失败：
1. 检查模型结构是否与训练时一致
2. 使用 strict=False 并检查缺失的键
3. 可能需要重新训练模型
""")

print("\n检查完成！")