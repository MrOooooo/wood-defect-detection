# models/iltm.py
"""
Instance-Linking Token Module (ILTM)
用于增强对模糊边界的敏感性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ILTM(nn.Module):
    """
    Instance-Linking Token Module
    使用可学习的token来表示不同的实例，通过相似度映射自适应增强特征
    """

    def __init__(self, num_tokens=100, feature_dim=768, rank=16, hidden_dim=512):
        super(ILTM, self).__init__()
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.rank = rank

        # 低秩分解生成可学习tokens: T = M × N
        self.token_M = nn.Parameter(torch.randn(num_tokens, rank))
        self.token_N = nn.Parameter(torch.randn(rank, feature_dim))

        # 用于投影特征和token的MLP
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.token_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # 用于融合的MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.token_M, std=0.02)
        nn.init.normal_(self.token_N, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        前向传播
        Args:
            features: (B, N, C) 输入特征，其中B是batch size，N是空间位置数，C是特征维度
        Returns:
            enhanced_features: (B, N, C) 增强后的特征
        """
        B, N, C = features.shape

        # 生成可学习tokens: T = M × N
        tokens = torch.matmul(self.token_M, self.token_N)  # (num_tokens, feature_dim)

        # 投影特征和tokens到共享嵌入空间
        F_p = self.feature_proj(features)  # (B, N, C)
        T_p = self.token_proj(tokens).unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, C)

        # 计算相似度映射: S = softmax(F_p · T_p^T)
        # (B, N, C) × (B, C, num_tokens) -> (B, N, num_tokens)
        similarity = torch.matmul(F_p, T_p.transpose(1, 2))
        similarity = F.softmax(similarity, dim=-1)  # (B, N, num_tokens)

        # Token加权特征: F_w = S · T
        # (B, N, num_tokens) × (B, num_tokens, C) -> (B, N, C)
        tokens_expanded = tokens.unsqueeze(0).expand(B, -1, -1)
        weighted_features = torch.matmul(similarity, tokens_expanded)

        # 通过MLP融合并添加残差连接
        fusion_output = self.fusion_mlp(weighted_features)
        enhanced_features = features + fusion_output

        return enhanced_features


if __name__ == "__main__":
    # 测试ILTM模块
    print("Testing ILTM Module...")

    # 创建模块
    iltm = ILTM(num_tokens=100, feature_dim=1280, rank=16)

    # 创建测试数据
    batch_size = 2
    num_patches = 256  # 例如16x16的patch
    feature_dim = 1280

    test_input = torch.randn(batch_size, num_patches, feature_dim)

    # 前向传播
    output = iltm(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in iltm.parameters())}")

    # 验证输出形状
    assert output.shape == test_input.shape, "Output shape mismatch!"
    print("ILTM Module test passed!")