# models/lsfa_pretrain.py
"""
LSFA: Self-supervised Feature Adaptation
✅ 最终修复版本 - 处理混合维度输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class LocalAdapter(nn.Module):
    """局部适配器"""

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # 如果输入是2D，扩展为3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, C) -> (B, 1, C)
            squeeze_output = True
        else:
            squeeze_output = False

        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        adapted = self.norm2(x + ffn_out)

        # 如果原始输入是2D，恢复为2D
        if squeeze_output:
            adapted = adapted.squeeze(1)  # (B, 1, C) -> (B, C)

        return adapted


class GlobalAdapter(nn.Module):
    """全局适配器"""

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        adapted = self.adapter(x)
        adapted = self.norm(x + adapted)
        return adapted


class LSFAModule(nn.Module):
    """LSFA模块"""

    def __init__(self, feature_dim=768, hidden_dim=512, num_heads=8, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature

        self.local_adapter = LocalAdapter(
            input_dim=feature_dim, hidden_dim=hidden_dim,
            output_dim=feature_dim, num_heads=num_heads
        )
        self.global_adapter = GlobalAdapter(
            input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=feature_dim
        )
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, local_features, global_features):
        adapted_local = self.local_adapter(local_features)
        adapted_global = self.global_adapter(global_features)
        return adapted_local, adapted_global

    def compute_contrastive_loss(self, features1, features2):
        z1 = F.normalize(self.projection_head(features1), dim=1)
        z2 = F.normalize(self.projection_head(features2), dim=1)

        B = z1.size(0)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        labels = torch.cat([
            torch.arange(B, device=z1.device) + B,
            torch.arange(B, device=z1.device)
        ], dim=0)

        mask = torch.eye(2 * B, device=z1.device).bool()
        similarity_matrix.masked_fill_(mask, -1e9)

        return F.cross_entropy(similarity_matrix, labels)


class LSFAPretrainer(nn.Module):
    """LSFA预训练器 - 最终版本"""

    def __init__(self, backbone, feature_dim=768, hidden_dim=512,
                 num_heads=8, temperature=0.07, adapt_layers=[8, 9, 10, 11]):
        super().__init__()
        self.backbone = backbone
        self.adapt_layers = adapt_layers
        self.feature_dim = feature_dim

        self.lsfa_modules = nn.ModuleList([
            LSFAModule(feature_dim, hidden_dim, num_heads, temperature)
            for _ in adapt_layers
        ])

        for param in self.backbone.parameters():
            param.requires_grad = False

    def _normalize_feature(self, feat):
        """
        统一特征格式为 (local_feat, global_feat)

        输入可能是:
        - (B, N, C): 3D patch特征
        - (B, C): 2D全局特征
        - tuple/list: 需要解包

        输出:
        - local_feat: (B, N, C) 或 (B, 1, C)
        - global_feat: (B, C)
        """
        # 如果是tuple或list，取第一个元素
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        # 确保是tensor
        if not isinstance(feat, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(feat)}")

        if feat.dim() == 3:
            # 3D特征: (B, N, C)
            B, N, C = feat.shape

            # 移除CLS token (如果存在)
            if N == 1297 or N == 257:
                feat_no_cls = feat[:, 1:, :]  # (B, N-1, C)
            else:
                feat_no_cls = feat

            # 计算全局特征
            global_feat = feat_no_cls.mean(dim=1)  # (B, C)
            local_feat = feat  # 保持原始3D格式

            return local_feat, global_feat

        elif feat.dim() == 2:
            # 2D特征: (B, C) - 已经是全局特征
            B, C = feat.shape

            # 为了适配LocalAdapter，创建伪3D特征
            local_feat = feat.unsqueeze(1)  # (B, 1, C)
            global_feat = feat  # (B, C)

            return local_feat, global_feat

        else:
            raise ValueError(f"Unexpected feature dimension: {feat.shape}")

    def forward(self, images, augmented_images):
        """前向传播"""
        with torch.no_grad():
            outputs_orig = self.backbone(images, output_hidden_states=True)
            outputs_aug = self.backbone(augmented_images, output_hidden_states=True)

        # ✅ 修复：从HuggingFace输出对象中提取hidden_states
        if hasattr(outputs_orig, 'hidden_states'):
            features_orig = outputs_orig.hidden_states
            features_aug = outputs_aug.hidden_states
        else:
            # 如果不是HF输出，假设是list/tuple
            features_orig = outputs_orig
            features_aug = outputs_aug

        # 🔍 调试
        # print(f"\n[DEBUG] Hidden states type: {type(features_orig)}")
        # print(f"[DEBUG] Num hidden states: {len(features_orig)}")
        # print(f"[DEBUG] First hidden state shape: {features_orig[0].shape}")

        total_local_loss = 0.0
        total_global_loss = 0.0
        adapted_features = []

        for i, layer_idx in enumerate(self.adapt_layers):
            # ✅ 修复：adapt_layers是逻辑层号[8,9,10,11]
            # hidden_states索引需要映射：layer_idx -> hidden_states[layer_idx+1]
            # 因为hidden_states[0]是embedding层
            hidden_idx = layer_idx + 1

            if hidden_idx >= len(features_orig):
                print(f"⚠️ Warning: layer {layer_idx} -> hidden_idx {hidden_idx} out of range")
                continue

            feat_orig = features_orig[hidden_idx]
            feat_aug = features_aug[hidden_idx]

            # if i == 0:
                # print(f"\n[DEBUG] Layer {layer_idx} (hidden_idx={hidden_idx}):")
                # print(f"  Input shape: {feat_orig.shape}")

            # 统一特征格式
            local_orig, global_orig = self._normalize_feature(feat_orig)
            local_aug, global_aug = self._normalize_feature(feat_aug)

            # LSFA适配
            adapted_local_orig, adapted_global_orig = self.lsfa_modules[i](
                local_orig, global_orig
            )
            adapted_local_aug, adapted_global_aug = self.lsfa_modules[i](
                local_aug, global_aug
            )

            # 对比损失
            local_loss = self.lsfa_modules[i].compute_contrastive_loss(
                adapted_global_orig, adapted_global_aug
            )
            global_loss = self.lsfa_modules[i].compute_contrastive_loss(
                adapted_global_orig, adapted_global_aug
            )

            total_local_loss += local_loss
            total_global_loss += global_loss

            # 恢复原始维度格式
            if feat_orig.dim() == 2:
                # 如果原始是2D，返回2D
                if adapted_local_orig.dim() == 3:
                    adapted_local_orig = adapted_local_orig.squeeze(1)

            adapted_features.append(adapted_local_orig)

        avg_local_loss = total_local_loss / len(self.adapt_layers)
        avg_global_loss = total_global_loss / len(self.adapt_layers)

        return {
            'total_loss': avg_local_loss + avg_global_loss,
            'local_loss': avg_local_loss,
            'global_loss': avg_global_loss,
            'adapted_features': adapted_features
        }

    def get_adapted_features(self, images):
        """获取适配后的特征"""
        with torch.no_grad():
            outputs = self.backbone(images, output_hidden_states=True)

        # ✅ 提取hidden_states
        if hasattr(outputs, 'hidden_states'):
            features = outputs.hidden_states
        else:
            features = outputs

        adapted_features = []
        for i, layer_idx in enumerate(self.adapt_layers):
            # ✅ 映射逻辑层号到hidden_states索引
            hidden_idx = layer_idx + 1

            if hidden_idx >= len(features):
                continue

            feat = features[hidden_idx]
            local_feat, global_feat = self._normalize_feature(feat)
            adapted_local, _ = self.lsfa_modules[i](local_feat, global_feat)

            # 恢复原始格式
            if feat.dim() == 2 and adapted_local.dim() == 3:
                adapted_local = adapted_local.squeeze(1)

            adapted_features.append(adapted_local)

        return adapted_features


class WoodTextureAugmentation:
    """数据增强"""

    def __init__(self):
        from torchvision import transforms as T

        self.strong_aug = T.Compose([
            T.RandomResizedCrop(512, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.weak_aug = T.Compose([
            T.RandomResizedCrop(512, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        return self.weak_aug(image), self.strong_aug(image)


if __name__ == "__main__":
    print("Testing LSFA - Final Version")


    class MixedBackbone(nn.Module):
        """模拟混合维度的backbone输出"""

        def forward(self, x, output_hidden_states=True):
            B = x.size(0)
            return [
                torch.randn(B, 1297, 768),  # Layer 8: 3D
                torch.randn(B, 768),  # Layer 9: 2D
                torch.randn(B, 768),  # Layer 10: 2D
                torch.randn(B, 1297, 768)  # Layer 11: 3D
            ]


    backbone = MixedBackbone()
    lsfa = LSFAPretrainer(backbone, feature_dim=768, adapt_layers=[8, 9, 10, 11])

    imgs = torch.randn(4, 3, 512, 512)
    aug_imgs = torch.randn(4, 3, 512, 512)

    outputs = lsfa(imgs, aug_imgs)
    print(f"✅ Total loss: {outputs['total_loss'].item():.4f}")
    print(f"✅ Features: {[f.shape for f in outputs['adapted_features']]}")