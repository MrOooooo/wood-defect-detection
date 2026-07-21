# models/lam_mask2former.py
"""
LAM + Mask2Former 完整模型

严格按照论文实现:
- DINOv2 Backbone (冻结)
- LAM 适配最后4层 (ILTM + FDM + LSM)
- Mask2Former Decoder 进行分割预测

架构: DINOv2 → LAM → Mask2Former Decoder → 分割结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .backbone import VFMBackbone
from .lam import MultiLayerLAM
from .mask2former_decoder import Mask2FormerDecoder


class LAMMask2FormerModel(nn.Module):
    """
    LAM + Mask2Former 语义分割模型

    论文架构:
    1. VFM Backbone (DINOv2) - 冻结参数
    2. LAM (ILTM + FDM + LSM) - 适配最后4层
    3. Mask2Former Decoder - 分割预测

    关键改进:
    - 使用Mask2Former的Transformer Decoder with Masked Attention
    - 多尺度特征融合
    - 100个可学习的object queries
    """

    def __init__(
            self,
            backbone_name='dinov2',
            num_classes=6,
            # LAM参数 (论文设置)
            num_tokens=100,
            token_rank=16,
            num_groups=16,
            use_lsm=True,
            tau=0.5,
            shared_tokens=True,
            adapt_layers=None,
            # Mask2Former参数
            hidden_dim=256,
            num_queries=100,
            num_decoder_layers=6,
            num_heads=8
    ):
        super(LAMMask2FormerModel, self).__init__()

        self.num_classes = num_classes

        # ========== 1. Backbone ==========
        print("\n" + "=" * 70)
        print("Initializing LAM + Mask2Former Model")
        print("=" * 70)

        self.backbone = VFMBackbone(
            model_name=backbone_name,
            freeze=True,
            output_layers=None
        )

        feature_dim = self.backbone.get_feature_dim()
        num_backbone_layers = self.backbone.get_num_layers()

        print(f"\nBackbone: {backbone_name}")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Total layers: {num_backbone_layers}")

        # ========== 2. 确定适配层 ==========
        if adapt_layers is None:
            if num_backbone_layers >= 4:
                adapt_layers = list(range(num_backbone_layers - 4, num_backbone_layers))
            else:
                adapt_layers = list(range(num_backbone_layers))

        valid_adapt_layers = [i for i in adapt_layers if 0 <= i < num_backbone_layers]
        if not valid_adapt_layers:
            valid_adapt_layers = list(range(max(0, num_backbone_layers - 4), num_backbone_layers))

        self.adapt_layers = valid_adapt_layers
        self.num_adapt_layers = len(valid_adapt_layers)
        self.backbone.output_layers = self.adapt_layers

        print(f"  Adapting layers: {self.adapt_layers}")

        # ========== 3. LAM ==========
        print(f"\nLAM Configuration:")
        print(f"  num_tokens (m): {num_tokens}")
        print(f"  token_rank (r): {token_rank}")
        print(f"  num_groups (G): {num_groups}")
        print(f"  tau: {tau}")
        print(f"  use_lsm: {use_lsm}")
        print(f"  shared_tokens: {shared_tokens}")

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

        # ========== 4. Mask2Former Decoder ==========
        print(f"\nMask2Former Decoder Configuration:")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  num_queries: {num_queries}")
        print(f"  num_decoder_layers: {num_decoder_layers}")
        print(f"  num_heads: {num_heads}")

        self.mask2former_decoder = Mask2FormerDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_feature_levels=self.num_adapt_layers
        )

        # 打印参数统计
        self._print_param_stats()

        print("\n" + "=" * 70)
        print("✅ Model initialization complete")
        print("=" * 70 + "\n")

    def _print_param_stats(self):
        """打印参数统计"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        lam_params = sum(p.numel() for p in self.multi_lam.parameters())
        decoder_params = sum(p.numel() for p in self.mask2former_decoder.parameters())
        total_params = backbone_params + lam_params + decoder_params

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nParameter Statistics:")
        print(f"  Backbone: {backbone_params:,} (frozen)")
        print(f"  LAM: {lam_params:,}")
        print(f"  Mask2Former Decoder: {decoder_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

    def forward(self, images, compute_cov_loss=False):
        """
        前向传播

        Args:
            images: (B, 3, H, W) 输入图像
            compute_cov_loss: 是否计算协方差损失
        Returns:
            logits: (B, num_classes, H, W) 语义分割logits
            cov_loss: 协方差损失 (如果compute_cov_loss=True)
        """
        B, _, H, W = images.shape

        # ========== 1. Backbone特征提取 ==========
        with torch.no_grad():
            features_list = self.backbone(images, output_hidden_states=True)

        # ========== 2. LAM特征增强 ==========
        if compute_cov_loss:
            enhanced_features, cov_loss = self.multi_lam(
                features_list,
                compute_cov_loss=True,
                training=self.training
            )
        else:
            enhanced_features = self.multi_lam(
                features_list,
                compute_cov_loss=False,
                training=self.training
            )
            cov_loss = None

        # ========== 3. 处理CLS token ==========
        # LAM输出包含CLS token, 需要移除
        processed_features = []
        for feat in enhanced_features:
            L = feat.shape[1]
            # 移除CLS token (第一个token)
            if L == 1297:  # 512x512 input: 36x36 + 1 CLS
                feat = feat[:, 1:, :]
            elif L == 257:  # 224x224 input: 16x16 + 1 CLS
                feat = feat[:, 1:, :]
            processed_features.append(feat)

        # ========== 4. Mask2Former Decoder ==========
        outputs = self.mask2former_decoder(processed_features, image_size=(H, W))

        # ========== 5. 转换为语义分割格式 ==========
        logits = self.mask2former_decoder.get_semantic_segmentation(outputs, (H, W))

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

        # Mask2Former Decoder参数
        for param in self.mask2former_decoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params

    def count_parameters(self):
        """统计参数量"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        lam_params = sum(p.numel() for p in self.multi_lam.parameters())
        lam_trainable = sum(p.numel() for p in self.multi_lam.parameters() if p.requires_grad)

        decoder_params = sum(p.numel() for p in self.mask2former_decoder.parameters())
        decoder_trainable = sum(p.numel() for p in self.mask2former_decoder.parameters() if p.requires_grad)

        total_params = backbone_params + lam_params + decoder_params
        total_trainable = backbone_trainable + lam_trainable + decoder_trainable

        return {
            'backbone_total': backbone_params,
            'backbone_trainable': backbone_trainable,
            'lam_total': lam_params,
            'lam_trainable': lam_trainable,
            'decoder_total': decoder_params,
            'decoder_trainable': decoder_trainable,
            'total': total_params,
            'trainable': total_trainable
        }


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("Testing LAM + Mask2Former Model")
    print("=" * 70)

    # 论文配置
    model = LAMMask2FormerModel(
        backbone_name='dinov2',
        num_classes=6,
        # LAM参数 (论文设置)
        num_tokens=100,  # m=100
        token_rank=16,  # r=16
        num_groups=16,  # G=16
        use_lsm=False,  # 第一阶段不使用LSM
        tau=0.5,
        shared_tokens=True,
        adapt_layers=[8, 9, 10, 11],  # 最后4层
        # Mask2Former参数
        hidden_dim=256,
        num_queries=100,
        num_decoder_layers=6
    )

    # 测试输入
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)

    print(f"\nInput: {images.shape}")

    # 训练模式测试
    model.train()
    logits, cov_loss = model(images, compute_cov_loss=True)

    print(f"\nTraining mode:")
    print(f"  Logits: {logits.shape}")
    print(f"  Cov loss: {cov_loss.item():.6f}")

    # 推理模式测试
    model.eval()
    with torch.no_grad():
        logits = model(images, compute_cov_loss=False)

    print(f"\nInference mode:")
    print(f"  Logits: {logits.shape}")

    # 验证输出
    assert logits.shape == (batch_size, 6, 512, 512), f"Expected (2, 6, 512, 512), got {logits.shape}"

    # 参数统计
    params = model.count_parameters()
    print(f"\nParameter Statistics:")
    print(f"  Backbone: {params['backbone_total']:,} (trainable: {params['backbone_trainable']:,})")
    print(f"  LAM: {params['lam_total']:,} (trainable: {params['lam_trainable']:,})")
    print(f"  Decoder: {params['decoder_total']:,} (trainable: {params['decoder_trainable']:,})")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
