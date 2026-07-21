# models/asl_module.py
"""
ASL (Arbitrary Self-supervised Learning) Module
整合到LAM框架中，提供patch-level对比学习能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchLevelContrastiveLoss(nn.Module):
    """
    Patch-level对比学习损失
    论文公式(9): L_s(i) = ||pred_ori_i - sg[pred_aug_i]|| + ||sg[pred_ori_i] - pred_aug_i||
    """

    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, feat_ori, feat_aug):
        """
        Args:
            feat_ori: (B, N, C) 原始特征
            feat_aug: (B, N, C) 增强特征
        Returns:
            loss: 标量
        """
        # Stop gradient
        loss = (
                F.l1_loss(feat_ori, feat_aug.detach()) +
                F.l1_loss(feat_ori.detach(), feat_aug)
        )
        return loss


class ASLFeatureAugmentation(nn.Module):
    """
    ASL特征增强模块
    实现两种设计：
    1. DLE (Dropout in Linear Embedding)
    2. AEE (Asymmetric structure at End of Encoder) - 默认
    """

    def __init__(self, feature_dim=768, aug_type='aee', dropout_rate=0.1):
        super().__init__()
        self.aug_type = aug_type

        if aug_type == 'dle':
            # Dropout in linear embedding
            self.dropout = nn.Dropout(p=dropout_rate)
        elif aug_type == 'aee':
            # Asymmetric Transformer layer
            self.aug_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown aug_type: {aug_type}")

    def forward(self, features):
        """
        Args:
            features: (B, N, C) 输入特征
        Returns:
            feat_ori: (B, N, C) 原始分支
            feat_aug: (B, N, C) 增强分支
        """
        if self.aug_type == 'dle':
            # 双路径：一路正常，一路dropout
            feat_ori = features
            feat_aug = self.dropout(features)
        elif self.aug_type == 'aee':
            # 双路径：一路直通，一路经过Transformer
            feat_ori = features
            feat_aug = self.aug_layer(features)

        return feat_ori, feat_aug


class ASLEnhancedILTM(nn.Module):
    """
    整合ASL的ILTM模块
    在原ILTM基础上增加patch-level对比学习
    """

    def __init__(
            self,
            num_tokens=100,
            feature_dim=768,
            rank=16,
            hidden_dim=512,
            use_asl=False,
            asl_aug_type='aee',
            asl_weight=1.0
    ):
        super().__init__()

        # 原始ILTM组件（保持不变）
        from .iltm import ILTM
        self.iltm = ILTM(num_tokens, feature_dim, rank, hidden_dim)

        # ASL组件（可选）
        self.use_asl = use_asl
        self.asl_weight = asl_weight

        if use_asl:
            self.feature_aug = ASLFeatureAugmentation(
                feature_dim=feature_dim,
                aug_type=asl_aug_type
            )
            self.patch_contrastive_loss = PatchLevelContrastiveLoss()

    def forward(self, features, compute_asl_loss=False):
        """
        Args:
            features: (B, N, C)
            compute_asl_loss: 是否计算ASL损失
        Returns:
            enhanced_features: (B, N, C)
            asl_loss: 标量（如果compute_asl_loss=True）
        """
        if not self.use_asl:
            # 不使用ASL，直接调用原ILTM
            enhanced = self.iltm(features)
            return enhanced if not compute_asl_loss else (enhanced, None)

        # ========== ASL增强流程 ==========
        # 1. 特征增强：生成双分支
        feat_ori, feat_aug = self.feature_aug(features)

        # 2. 通过ILTM处理两个分支
        enhanced_ori = self.iltm(feat_ori)
        enhanced_aug = self.iltm(feat_aug)

        # 3. 计算patch-level对比损失
        asl_loss = None
        if compute_asl_loss and self.training:
            asl_loss = self.patch_contrastive_loss(enhanced_ori, enhanced_aug)
            asl_loss = asl_loss * self.asl_weight

        # 4. 返回主分支的增强特征
        if compute_asl_loss:
            return enhanced_ori, asl_loss
        else:
            return enhanced_ori


if __name__ == "__main__":
    print("Testing ASL-Enhanced ILTM...")

    # 测试参数
    batch_size = 2
    num_patches = 256
    feature_dim = 768

    # 创建模块（启用ASL）
    asl_iltm = ASLEnhancedILTM(
        num_tokens=100,
        feature_dim=feature_dim,
        use_asl=True,
        asl_aug_type='aee',
        asl_weight=1.0
    )

    # 测试数据
    test_input = torch.randn(batch_size, num_patches, feature_dim)

    # 训练模式
    asl_iltm.train()
    enhanced, asl_loss = asl_iltm(test_input, compute_asl_loss=True)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {enhanced.shape}")
    print(f"ASL loss: {asl_loss.item():.6f}")

    # 推理模式
    asl_iltm.eval()
    with torch.no_grad():
        enhanced = asl_iltm(test_input, compute_asl_loss=False)

    print(f"\nInference output shape: {enhanced.shape}")
    print("✅ Test passed!")