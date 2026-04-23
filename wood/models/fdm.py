
"""
Feature Disentanglement Module (FDM)
用于减少特征冗余，提高特征独立性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FDM(nn.Module):
    """
    Feature Disentanglement Module
    通过组卷积和通道混洗减少类间相似性和特征纠缠
    """

    def __init__(self, feature_dim=768, num_groups=16):
        super(FDM, self).__init__()
        self.feature_dim = feature_dim

        # 确保feature_dim能被num_groups整除
        self.num_groups = self._find_valid_num_groups(feature_dim, num_groups)
        if self.num_groups != num_groups:
            print(f"Warning: Adjusted num_groups from {num_groups} to {self.num_groups} "
                  f"to match feature_dim {feature_dim}")

        self.channels_per_group = feature_dim // self.num_groups

        # 组卷积
        self.group_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
            groups=self.num_groups,
            bias=False
        )

        # Batch Normalization
        self.bn = nn.BatchNorm1d(feature_dim)

        # 激活函数
        self.activation = nn.GELU()

        self._init_weights()

    def _find_valid_num_groups(self, feature_dim, desired_groups):
        """
        找到能整除feature_dim的最接近desired_groups的数
        """
        # 如果desired_groups能整除，直接返回
        if feature_dim % desired_groups == 0:
            return desired_groups

        # 否则找最接近的因数
        best_groups = 1
        min_diff = abs(desired_groups - 1)

        # 从desired_groups向下搜索
        for g in range(desired_groups, 0, -1):
            if feature_dim % g == 0:
                if abs(g - desired_groups) < min_diff:
                    best_groups = g
                    min_diff = abs(g - desired_groups)
                break

        # 也向上搜索一下
        for g in range(desired_groups + 1, feature_dim + 1):
            if feature_dim % g == 0:
                if abs(g - desired_groups) < min_diff:
                    best_groups = g
                break

        return best_groups

    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_normal_(self.group_conv.weight, mode='fan_out', nonlinearity='relu')

    def channel_shuffle(self, x, groups):
        """
        通道混洗操作
        Args:
            x: (B, C, N) 输入特征
            groups: 组数
        Returns:
            shuffled: (B, C, N) 混洗后的特征
        """
        B, C, N = x.shape
        channels_per_group = C // groups

        # reshape
        x = x.view(B, groups, channels_per_group, N)

        # transpose
        x = x.transpose(1, 2).contiguous()

        # flatten
        x = x.view(B, C, N)

        return x

    def compute_covariance_loss(self, features):
        """
        计算协方差矩阵正则化损失
        Args:
            features: (B, C, N) 特征
        Returns:
            loss: 标量，协方差损失
        """
        B, C, N = features.shape

        # 将特征按组分割
        features = features.view(B, self.num_groups, self.channels_per_group, N)

        total_loss = 0.0

        for i in range(self.num_groups):
            # 获取第i组特征: (B, channels_per_group, N)
            group_feature = features[:, i, :, :]

            for b in range(B):
                feat = group_feature[b]  # (channels_per_group, N)

                # 中心化
                feat_mean = feat.mean(dim=1, keepdim=True)
                feat_centered = feat - feat_mean

                # 计算协方差矩阵
                cov = torch.matmul(feat_centered, feat_centered.t()) / N

                # 计算与单位矩阵的Frobenius范数
                identity = torch.eye(self.channels_per_group, device=cov.device)
                loss = torch.norm(cov - identity, p='fro') ** 2
                total_loss += loss

        # 平均损失
        total_loss = total_loss / (B * self.num_groups)

        return total_loss

    def forward(self, features, compute_loss=False):
        """
        前向传播
        Args:
            features: (B, N, C) 输入特征
            compute_loss: 是否计算协方差损失
        Returns:
            output: (B, N, C) 解耦后的特征
            cov_loss: 协方差损失（如果compute_loss=True）
        """
        B, N, C = features.shape

        # 检查特征维度是否匹配
        if C != self.feature_dim:
            # 动态调整（但不应该在运行时发生）
            print(f"Warning: Feature dim mismatch! Expected {self.feature_dim}, got {C}")
            # 使用projection调整维度
            if not hasattr(self, 'proj'):
                self.proj = nn.Linear(C, self.feature_dim).to(features.device)
            features_proj = self.proj(features)
            x = features_proj.transpose(1, 2)  # (B, feature_dim, N)
        else:
            # 转换为 (B, C, N) 以适配Conv1d
            x = features.transpose(1, 2)  # (B, C, N)

        # 组卷积
        x = self.group_conv(x)  # (B, C, N)
        x = self.bn(x)
        x = self.activation(x)

        # 通道混洗
        x = self.channel_shuffle(x, self.num_groups)

        # 计算协方差损失（仅在训练时）
        cov_loss = None
        if compute_loss:
            cov_loss = self.compute_covariance_loss(x)

        # 转换回 (B, N, C)
        output = x.transpose(1, 2)

        if compute_loss:
            return output, cov_loss
        else:
            return output


if __name__ == "__main__":
    # 测试FDM模块
    print("Testing FDM Module...")

    # 测试不同的特征维度
    for feature_dim in [768, 1024, 1280]:
        print(f"\n Testing with feature_dim={feature_dim}")

        # 创建模块
        fdm = FDM(feature_dim=feature_dim, num_groups=16)

        # 创建测试数据
        batch_size = 2
        num_patches = 256

        test_input = torch.randn(batch_size, num_patches, feature_dim)

        # 测试不计算损失
        output = fdm(test_input, compute_loss=False)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Num groups: {fdm.num_groups}")
        print(f"  Channels per group: {fdm.channels_per_group}")

        # 测试计算损失
        output, cov_loss = fdm(test_input, compute_loss=True)
        print(f"  Covariance loss: {cov_loss.item():.4f}")

        # 验证输出形状
        assert output.shape == test_input.shape, "Output shape mismatch!"

    print("\n✓ FDM Module test passed!")