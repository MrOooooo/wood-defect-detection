# models/lsm.py
"""
Layer Switch Module (LSM)
用于动态决定是否激活LAM模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSM(nn.Module):
    """
    Layer Switch Module
    使用通道注意力和Gumbel-Softmax动态决定是否激活LAM
    """

    def __init__(self, feature_dim=768, reduction_ratio=16, tau=0.5):
        super(LSM, self).__init__()
        self.feature_dim = feature_dim
        self.reduction_ratio = reduction_ratio
        self.tau = tau  # Gumbel-Softmax温度参数

        hidden_dim = feature_dim // reduction_ratio

        # 通道注意力网络
        self.channel_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

        # 用于捕获局部空间相关性的卷积
        self.spatial_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim  # 深度可分离卷积
        )

        # 二分类器：决定是否激活LAM
        self.classifier = nn.Linear(feature_dim, 2)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def gumbel_softmax(self, logits, tau=1.0, hard=False):
        """
        Gumbel-Softmax采样
        Args:
            logits: (B, 2) 未归一化的logits
            tau: 温度参数
            hard: 是否使用硬决策（straight-through estimator）
        Returns:
            y: (B, 2) Gumbel-Softmax输出
        """
        # 采样Gumbel噪声
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim=-1)

        if hard:
            # Straight-through estimator
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            # 前向使用hard，反向使用soft
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft

        return y

    def forward(self, features, training=True):
        """
        前向传播
        Args:
            features: (B, N, C) 输入特征
            training: 是否处于训练模式
        Returns:
            switch: (B,) 二值开关信号，1表示激活LAM，0表示跳过
            probs: (B, 2) 激活概率（用于可视化和分析）
        """
        B, N, C = features.shape

        # 全局平均池化
        gap = features.mean(dim=1)  # (B, C)

        # 通道注意力
        attention_weights = self.channel_attention(gap)  # (B, C)

        # 应用注意力权重
        weighted_features = features * attention_weights.unsqueeze(1)  # (B, N, C)

        # 残差连接
        weighted_features = weighted_features + features

        # 转换为 (B, C, N) 以适配Conv1d
        x = weighted_features.transpose(1, 2)  # (B, C, N)

        # 空间卷积捕获局部相关性
        x = self.spatial_conv(x)  # (B, C, N)

        # 转换回 (B, N, C) 并展平
        x = x.transpose(1, 2)  # (B, N, C)
        x = x.reshape(B, -1)  # (B, N*C)

        # 全局平均池化以减少维度
        x = x.mean(dim=1, keepdim=True)  # (B, 1)
        x = x.expand(-1, C)  # (B, C)

        # 二分类
        logits = self.classifier(x)  # (B, 2)

        # 计算概率
        probs = F.softmax(logits, dim=-1)  # (B, 2)

        if training:
            # 训练时使用Gumbel-Softmax
            y = self.gumbel_softmax(logits, tau=self.tau, hard=True)
        else:
            # 推理时使用argmax
            y = torch.zeros_like(probs)
            y.scatter_(1, probs.argmax(dim=1, keepdim=True), 1.0)

        # 提取开关信号：y[:, 0]表示ON，y[:, 1]表示OFF
        switch = y[:, 0]  # (B,)

        print(f"LSM Switch values:{switch}")

        return switch, probs


if __name__ == "__main__":
    # 测试LSM模块
    print("Testing LSM Module...")

    # 创建模块
    lsm = LSM(feature_dim=1280, reduction_ratio=16, tau=0.5)

    # 创建测试数据
    batch_size = 4
    num_patches = 256
    feature_dim = 1280

    test_input = torch.randn(batch_size, num_patches, feature_dim)

    # 训练模式
    lsm.train()
    switch, probs = lsm(test_input, training=True)

    print(f"Input shape: {test_input.shape}")
    print(f"Switch shape: {switch.shape}")
    print(f"Switch values: {switch}")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs: {probs}")

    # 推理模式
    lsm.eval()
    switch, probs = lsm(test_input, training=False)
    print(f"\nInference mode:")
    print(f"Switch values: {switch}")
    print(f"Probs: {probs}")

    print(f"\nNumber of parameters: {sum(p.numel() for p in lsm.parameters())}")
    print("LSM Module test passed!")