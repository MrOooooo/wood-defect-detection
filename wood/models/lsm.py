"""
Layer Switch Module (LSM)
MODIFIED: preserve channel-aware spatial aggregation before binary switching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSM(nn.Module):
    def __init__(self, feature_dim=768, reduction_ratio=16, tau=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduction_ratio = reduction_ratio
        self.tau = tau

        hidden_dim = feature_dim // reduction_ratio

        self.channel_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        self.spatial_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim,
        )

        self.classifier = nn.Linear(feature_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def gumbel_softmax(self, logits, tau=1.0, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def forward(self, features, training=True):
        batch_size, _, channels = features.shape

        gap = features.mean(dim=1)
        attention_weights = self.channel_attention(gap)
        weighted_features = features * attention_weights.unsqueeze(1)
        weighted_features = weighted_features + features

        x = weighted_features.transpose(1, 2)
        x = self.spatial_conv(x)

        # MODIFIED: spatial aggregation now keeps per-channel structure for the classifier.
        x = x.mean(dim=2).reshape(batch_size, channels)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)

        if training:
            y = self.gumbel_softmax(logits, tau=self.tau, hard=True)
        else:
            y = torch.zeros_like(probs)
            y.scatter_(1, probs.argmax(dim=1, keepdim=True), 1.0)

        switch = y[:, 0]
        return switch, probs
