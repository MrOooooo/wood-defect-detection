# debug_backbone.py
"""
调试脚本：检查backbone特征和层数
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import VFMBackbone


def debug_backbone():
    """调试backbone输出"""
    print("=" * 60)
    print("Debugging DINOv2 Backbone")
    print("=" * 60)

    # 创建backbone
    backbone = VFMBackbone(
        model_name='dinov2',
        freeze=True,
        output_layers=None
    )

    print(f"\nBackbone Info:")
    print(f"  - Feature dim: {backbone.get_feature_dim()}")
    print(f"  - Num layers: {backbone.get_num_layers()}")
    print(f"  - Patch size: {backbone.patch_size}")

    # 测试不同尺寸的输入
    test_sizes = [224, 384, 512]

    for size in test_sizes:
        print(f"\n{'=' * 60}")
        print(f"Testing with image size: {size}x{size}")
        print(f"{'=' * 60}")

        # 创建测试输入
        test_input = torch.randn(1, 3, size, size)

        # 提取特征
        with torch.no_grad():
            features_list = backbone(test_input, output_hidden_states=True)

        print(f"\nNumber of layer outputs: {len(features_list)}")

        for i, feat in enumerate(features_list):
            B, L, C = feat.shape
            print(f"  Layer {i:2d}: shape={feat.shape}, L={L}, C={C}")

            # 计算对应的patch grid
            if L > 1:
                # 可能有CLS token
                L_no_cls = L - 1
                grid_size = int(L_no_cls ** 0.5)
                print(f"           -> Grid size (if has CLS): {grid_size}x{grid_size}")

                # 或者没有CLS token
                grid_size_no_cls = int(L ** 0.5)
                print(f"           -> Grid size (if no CLS): {grid_size_no_cls}x{grid_size_no_cls}")

        # 检查最后一层特征
        last_feat = features_list[-1]
        print(f"\n  Last layer feature shape: {last_feat.shape}")
        print(f"  Expected patch grid for {size}x{size} image with patch_size={backbone.patch_size}:")
        expected_grid = size // backbone.patch_size
        print(f"    {expected_grid}x{expected_grid} = {expected_grid ** 2} patches")
        print(f"    With CLS token: {expected_grid ** 2 + 1}")

    # 测试adapt_layers
    print(f"\n{'=' * 60}")
    print("Testing adapt_layers selection")
    print(f"{'=' * 60}")

    num_layers = backbone.get_num_layers()

    # 测试不同的adapt_layers配置
    test_configs = [
        None,  # 自动选择
        [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1],  # 最后4层
        [num_layers - 2, num_layers - 1],  # 最后2层
        [num_layers - 1],  # 只有最后1层
    ]

    for i, config in enumerate(test_configs):
        print(f"\nConfig {i + 1}: adapt_layers = {config}")

        if config is None:
            # 自动选择最后4层
            if num_layers >= 4:
                selected = list(range(num_layers - 4, num_layers))
            else:
                selected = list(range(num_layers))
        else:
            selected = config

        print(f"  -> Will adapt layers: {selected}")
        print(f"  -> Number of LAM modules needed: {len(selected)}")


def test_model_creation():
    """测试模型创建"""
    print(f"\n{'=' * 60}")
    print("Testing Model Creation")
    print(f"{'=' * 60}")

    from models import LAMSegmentationModel

    try:
        model = LAMSegmentationModel(
            backbone_name='dinov2',
            num_classes=4,
            num_tokens=64,
            token_rank=32,
            num_groups=8,
            use_lsm=False,
            tau=1.0,
            shared_tokens=True,
            adapt_layers=None  # 自动选择
        )

        print("\n✓ Model created successfully!")

        # 测试前向传播
        test_input = torch.randn(2, 3, 512, 512)

        print("\nTesting forward pass...")
        with torch.no_grad():
            logits, cov_loss = model(test_input, compute_cov_loss=True)

        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Cov loss: {cov_loss.item():.4f}")

        # 统计参数
        param_count = model.count_parameters()
        print(f"\nParameter count:")
        print(f"  LAM trainable: {param_count['lam_trainable']:,}")
        print(f"  Head trainable: {param_count['head_trainable']:,}")
        print(f"  Total trainable: {param_count['trainable']:,}")

    except Exception as e:
        print(f"\n✗ Model creation failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_backbone()
    test_model_creation()

    print(f"\n{'=' * 60}")
    print("Debug completed!")
    print(f"{'=' * 60}")