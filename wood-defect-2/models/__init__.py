# models/__init__.py
"""
完整的LAM分割模型 - 修复版本
"""

import torch
import torch.nn as nn
import math
from .backbone import VFMBackbone, SegmentationHead
from .lam import MultiLayerLAM


class LAMSegmentationModel(nn.Module):
    def __init__(self, backbone_name='dinov2', num_classes=5, num_tokens=100,
                 token_rank=16, num_groups=16, use_lsm=True, tau=0.5,
                 shared_tokens=True, adapt_layers=None):
        super(LAMSegmentationModel, self).__init__()

        # 加载backbone
        self.backbone = VFMBackbone(
            model_name=backbone_name,
            freeze=True,
            output_layers=None
        )

        feature_dim = self.backbone.get_feature_dim()
        num_backbone_layers = self.backbone.get_num_layers()

        print(f"Backbone info:")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Total layers: {num_backbone_layers}")

        # 默认适配最后4层
        if adapt_layers is None:
            if num_backbone_layers >= 4:
                adapt_layers = list(range(num_backbone_layers - 4, num_backbone_layers))
            else:
                adapt_layers = list(range(num_backbone_layers))

        # **关键修复:验证adapt_layers**
        valid_adapt_layers = [i for i in adapt_layers if 0 <= i < num_backbone_layers]
        if not valid_adapt_layers:
            print(f"Warning: No valid adapt_layers, using last 4 layers")
            valid_adapt_layers = list(range(max(0, num_backbone_layers - 4), num_backbone_layers))

        self.adapt_layers = valid_adapt_layers
        self.num_adapt_layers = len(valid_adapt_layers)
        print(f"  - Adapting layers: {self.adapt_layers}")

        # 更新backbone的output_layers
        self.backbone.output_layers = self.adapt_layers

        # 创建LAM
        self.multi_lam = MultiLayerLAM(
            num_layers=self.num_adapt_layers,
            feature_dim=feature_dim,
            num_tokens=num_tokens,
            token_rank=token_rank,
            num_groups=num_groups,
            use_lsm=use_lsm,
            tau=tau,
            shared_tokens=shared_tokens
        )

        # 分割头
        self.seg_head = SegmentationHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=256
        )

        self.num_classes = num_classes

    def forward(self, images, compute_cov_loss=False):
        """
        前向传播
        Args:
            images: (B, 3, H, W) 输入图像
            compute_cov_loss: 是否计算协方差损失
        Returns:
            logits: (B, num_classes, H, W) 分割预测
            cov_loss: 协方差损失（如果compute_cov_loss=True）
        """
        B, _, H, W = images.shape

        # 提取特征
        with torch.no_grad():
            features_list = self.backbone(images, output_hidden_states=True)

        # if len(features_list) < self.num_adapt_layers:
        #     print(f"Warning: Got {len(features_list)} features but expected {self.num_adapt_layers}")
        #     # 如果特征不足，使用可用的特征
        #     selected_features = features_list
        # else:
        #     # 只选择需要适配的层
        #     selected_features = []
        #     for i in self.adapt_layers:
        #         if i < len(features_list):
        #             selected_features.append(features_list[i])
        #         else:
        #             print(f"Warning: Layer {i} out of range, using last layer")
        #             selected_features.append(features_list[-1])

        # features_list 已经是按 adapt_layers 筛选过的，直接使用即可
        selected_features = features_list  # 不需要再次筛选！

        # 通过LAM增强特征
        if compute_cov_loss:
            enhanced_features, cov_loss = self.multi_lam(
                selected_features,
                compute_cov_loss=True,
                training=self.training
            )
        else:
            enhanced_features = self.multi_lam(
                selected_features,
                compute_cov_loss=False,
                training=self.training
            )
            cov_loss = None

        # 使用最后一层的增强特征进行分割
        final_features = enhanced_features[-1]  # (B, L, C)

        # 移除[CLS] token（如果存在）
        L = final_features.shape[1]

        # 尝试检测是否有CLS token
        # DINOv2通常第一个token是CLS
        if L == 257:  # 16x16 patches + 1 CLS for 224x224 input
            final_features = final_features[:, 1:, :]
        elif L == 1297:  # 36x36 patches + 1 CLS for 512x512 input
            final_features = final_features[:, 1:, :]

        # 重新计算L
        L = final_features.shape[1]

        # 计算合适的patch网格尺寸
        patch_grid_h = int(math.sqrt(L))
        patch_grid_w = patch_grid_h

        # 如果不是完全平方数，调整
        while patch_grid_h * patch_grid_w < L:
            patch_grid_w += 1

        # 如果需要，截断或填充特征
        target_L = patch_grid_h * patch_grid_w
        if target_L != L:
            if target_L > L:
                # 填充
                padding = target_L - L
                final_features = torch.nn.functional.pad(
                    final_features, (0, 0, 0, padding), mode='constant', value=0
                )
            else:
                # 截断
                final_features = final_features[:, :target_L, :]

        # 生成分割mask
        logits = self.seg_head(final_features, image_size=(H, W))

        if compute_cov_loss:
            return logits, cov_loss
        else:
            return logits

    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []

        # LAM参数
        for param in self.multi_lam.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # 分割头参数
        for param in self.seg_head.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params

    def count_parameters(self):
        """统计参数量"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )

        lam_params = sum(p.numel() for p in self.multi_lam.parameters())
        lam_trainable = sum(
            p.numel() for p in self.multi_lam.parameters() if p.requires_grad
        )

        head_params = sum(p.numel() for p in self.seg_head.parameters())
        head_trainable = sum(
            p.numel() for p in self.seg_head.parameters() if p.requires_grad
        )

        total_params = backbone_params + lam_params + head_params
        total_trainable = backbone_trainable + lam_trainable + head_trainable

        return {
            'backbone_total': backbone_params,
            'backbone_trainable': backbone_trainable,
            'lam_total': lam_params,
            'lam_trainable': lam_trainable,
            'head_total': head_params,
            'head_trainable': head_trainable,
            'total': total_params,
            'trainable': total_trainable
        }


if __name__ == "__main__":
    print("Testing LAM Segmentation Model...")

    model = LAMSegmentationModel(
        backbone_name='dinov2',
        num_classes=5,
        num_tokens=100,
        token_rank=16,
        num_groups=16,
        use_lsm=True,
        adapt_layers=None  # 自动选择最后4层
    )

    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)

    model.train()
    logits, cov_loss = model(images, compute_cov_loss=True)

    print(f"Input shape: {images.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Covariance loss: {cov_loss.item()}")

    model.eval()
    with torch.no_grad():
        logits = model(images, compute_cov_loss=False)

    print(f"Inference logits shape: {logits.shape}")

    param_count = model.count_parameters()

    print("\nParameter Statistics:")
    print(f"  Backbone:")
    print(f"    Total: {param_count['backbone_total']:,}")
    print(f"    Trainable: {param_count['backbone_trainable']:,}")
    print(f"  LAM:")
    print(f"    Total: {param_count['lam_total']:,}")
    print(f"    Trainable: {param_count['lam_trainable']:,}")
    print(f"  Segmentation Head:")
    print(f"    Total: {param_count['head_total']:,}")
    print(f"    Trainable: {param_count['head_trainable']:,}")
    print(f"  Overall:")
    print(f"    Total: {param_count['total']:,}")
    print(f"    Trainable: {param_count['trainable']:,}")

    trainable_ratio = param_count['trainable'] / param_count['total'] * 100
    print(f"    Trainable Ratio: {trainable_ratio:.2f}%")

    print("\nModel test passed!")