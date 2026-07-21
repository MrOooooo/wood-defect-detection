# models/mask2former_decoder.py
"""
Mask2Former Decoder - 用于接收LAM增强特征的语义分割头

严格按照论文实现：
- 使用Transformer Decoder with Masked Attention
- 多尺度特征融合
- 100个可学习的object queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEmbeddingSine(nn.Module):
    """2D正弦位置编码"""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            pos: (B, num_pos_feats*2, H, W)
        """
        B, _, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H + eps) * self.scale
            x_embed = x_embed / (W + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)

        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)  # (C, H, W)
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, C, H, W)

        return pos


class MultiScalePixelDecoder(nn.Module):
    """
    多尺度像素解码器
    将LAM的多层特征转换为多尺度特征金字塔
    """

    def __init__(self, feature_dim=768, hidden_dim=256, num_feature_levels=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        # 输入投影层 - 每层LAM特征一个
        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ) for _ in range(num_feature_levels)
        ])

        # 特征融合层
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_feature_levels)
        ])

        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 多尺度特征的位置编码
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: list of (B, H*W, C), 4层LAM输出
        Returns:
            mask_features: (B, hidden_dim, H, W) 用于mask预测
            multi_scale_features: list of (B, hidden_dim, H, W) 多尺度特征
            pos_encodings: list of (B, hidden_dim, H, W) 位置编码
        """
        B = multi_scale_features[0].shape[0]

        # 推断空间尺寸
        L = multi_scale_features[0].shape[1]
        H = W = int(math.sqrt(L))

        # 投影并reshape每层特征
        projected_features = []
        for i, (feat, proj) in enumerate(zip(multi_scale_features, self.input_projections)):
            # (B, L, C) -> (B, C, H, W)
            feat_2d = feat.transpose(1, 2).reshape(B, self.feature_dim, H, W)
            feat_proj = proj(feat_2d)
            projected_features.append(feat_proj)

        # 自顶向下的特征融合 (FPN-style)
        # 从最深层开始
        fused_features = [None] * self.num_feature_levels
        fused_features[-1] = self.lateral_convs[-1](projected_features[-1])

        for i in range(self.num_feature_levels - 2, -1, -1):
            # 上采样深层特征
            upsampled = F.interpolate(
                fused_features[i + 1],
                size=projected_features[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            # 融合
            fused_features[i] = self.lateral_convs[i](projected_features[i] + upsampled)

        # 最终mask features (使用最高分辨率)
        mask_features = self.output_conv(fused_features[0])

        # 计算位置编码
        pos_encodings = [self.pe_layer(feat) for feat in fused_features]

        return mask_features, fused_features, pos_encodings


class MaskedAttention(nn.Module):
    """
    Mask2Former的核心: Masked Cross-Attention
    只在预测的前景区域内计算attention
    """

    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (B, num_queries, C)
            key: (B, H*W, C)
            value: (B, H*W, C)
            attn_mask: (B, num_queries, H*W) 可选的attention mask
        Returns:
            output: (B, num_queries, C)
        """
        B, N_q, _ = query.shape
        _, N_kv, _ = key.shape

        # 投影
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算attention scores
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, N_q, N_kv)

        # 应用mask (Mask2Former的核心)
        if attn_mask is not None:
            # attn_mask: (B, N_q, N_kv) -> (B, 1, N_q, N_kv)
            attn_mask = attn_mask.unsqueeze(1)
            # mask为True的位置设为-inf
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        output = torch.matmul(attn, v)  # (B, heads, N_q, head_dim)
        output = output.transpose(1, 2).reshape(B, N_q, self.hidden_dim)
        output = self.out_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """Mask2Former Transformer Decoder层"""

    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=2048, dropout=0.0):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Masked cross-attention
        self.cross_attn = MaskedAttention(hidden_dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, memory, memory_pos, attn_mask=None):
        """
        Args:
            query: (B, num_queries, C)
            memory: (B, H*W, C) 像素特征
            memory_pos: (B, H*W, C) 位置编码
            attn_mask: (B, num_queries, H*W) mask attention的mask
        Returns:
            query: (B, num_queries, C)
        """
        # Self-attention
        q = self.self_attn_norm(query)
        q2, _ = self.self_attn(q, q, q)
        query = query + q2

        # Masked cross-attention
        q = self.cross_attn_norm(query)
        # 添加位置编码到key
        key = memory + memory_pos
        q2 = self.cross_attn(q, key, memory, attn_mask)
        query = query + q2

        # FFN
        q = self.ffn_norm(query)
        query = query + self.ffn(q)

        return query


class Mask2FormerDecoder(nn.Module):
    """
    完整的Mask2Former Decoder
    接收LAM增强的多层特征，输出语义分割结果
    """

    def __init__(
            self,
            feature_dim=768,
            hidden_dim=256,
            num_classes=6,
            num_queries=100,
            num_decoder_layers=6,
            num_heads=8,
            num_feature_levels=4
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # Pixel Decoder
        self.pixel_decoder = MultiScalePixelDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_feature_levels=num_feature_levels
        )

        # 可学习的object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        # Transformer Decoder Layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * 4,
                dropout=0.0
            ) for _ in range(num_decoder_layers)
        ])

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Prediction Heads
        # Class prediction head
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object

        # Mask prediction head (MLP)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Level embedding for multi-scale
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # 初始化query embeddings
        nn.init.normal_(self.query_embed.weight)
        nn.init.normal_(self.query_feat.weight)

        # 初始化class embed - bias设置使得初始时倾向于预测no-object
        nn.init.constant_(self.class_embed.bias, 0)
        nn.init.constant_(self.class_embed.bias[-1], -2.0)  # no-object bias

    def forward(self, multi_scale_features, image_size):
        """
        Args:
            multi_scale_features: list of (B, L, C), 4层LAM输出 (移除了CLS token)
            image_size: (H, W) 目标输出尺寸
        Returns:
            outputs: dict with 'pred_logits' and 'pred_masks'
        """
        B = multi_scale_features[0].shape[0]

        # 1. Pixel Decoder - 获取多尺度特征
        mask_features, fused_features, pos_encodings = self.pixel_decoder(multi_scale_features)

        # 2. 准备decoder输入
        # 初始化queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, C)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, C)

        # 将多尺度特征展平并拼接
        # 使用最高分辨率的特征作为memory
        H, W = fused_features[0].shape[-2:]
        memory = fused_features[0].flatten(2).transpose(1, 2)  # (B, H*W, C)
        memory_pos = pos_encodings[0].flatten(2).transpose(1, 2)  # (B, H*W, C)

        # 3. Transformer Decoder
        output = query_feat
        predictions_class = []
        predictions_mask = []

        for i, layer in enumerate(self.decoder_layers):
            # 计算当前的attention mask
            # 基于上一轮的mask预测
            if i == 0:
                attn_mask = None
            else:
                # 使用上一轮预测的mask作为attention mask
                # mask_pred: (B, num_queries, H, W)
                mask_pred = predictions_mask[-1]
                # Threshold并创建mask
                attn_mask = (mask_pred.flatten(2) < 0).detach()  # (B, num_queries, H*W)

            output = layer(output, memory, memory_pos, attn_mask)

            # 中间预测
            class_pred = self.class_embed(self.decoder_norm(output))  # (B, num_queries, num_classes+1)
            mask_embed = self.mask_embed(self.decoder_norm(output))  # (B, num_queries, C)

            # Mask预测: query embeddings与mask features做点积
            # mask_features: (B, C, H, W)
            mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)

            predictions_class.append(class_pred)
            predictions_mask.append(mask_pred)

        # 4. 最终输出
        # 使用最后一层的预测
        final_class = predictions_class[-1]  # (B, num_queries, num_classes+1)
        final_mask = predictions_mask[-1]  # (B, num_queries, H, W)

        # 上采样mask到目标尺寸
        final_mask = F.interpolate(
            final_mask,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )

        return {
            'pred_logits': final_class,
            'pred_masks': final_mask,
            'aux_outputs': [
                {'pred_logits': c, 'pred_masks': m}
                for c, m in zip(predictions_class[:-1], predictions_mask[:-1])
            ]
        }

    def get_semantic_segmentation(self, outputs, image_size):
        """
        将Mask2Former输出转换为语义分割格式

        Args:
            outputs: forward的输出
            image_size: (H, W)
        Returns:
            semantic_logits: (B, num_classes, H, W)
        """
        pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes+1)
        pred_masks = outputs['pred_masks']  # (B, num_queries, H, W)

        # 确保mask尺寸正确
        if pred_masks.shape[-2:] != image_size:
            pred_masks = F.interpolate(
                pred_masks,
                size=image_size,
                mode='bilinear',
                align_corners=False
            )

        B, Q, H, W = pred_masks.shape

        # 类别概率 (排除no-object类)
        class_probs = F.softmax(pred_logits, dim=-1)[:, :, :-1]  # (B, Q, num_classes)

        # Mask概率
        mask_probs = pred_masks.sigmoid()  # (B, Q, H, W)

        # 语义分割: 对每个像素，选择得分最高的query-class组合
        # 方法1: 加权聚合
        semantic_logits = torch.einsum('bqc,bqhw->bchw', class_probs, mask_probs)

        return semantic_logits


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Mask2Former Decoder with LAM Features")
    print("=" * 70)

    # 测试参数
    batch_size = 2
    feature_dim = 768
    num_classes = 6
    image_size = (512, 512)

    # 模拟LAM输出 (4层, 移除了CLS token)
    # 512/14 ≈ 36, 36x36 = 1296 patches
    num_patches = 1296
    multi_scale_features = [
        torch.randn(batch_size, num_patches, feature_dim)
        for _ in range(4)
    ]

    print(f"\nInput (4 LAM layers):")
    for i, feat in enumerate(multi_scale_features):
        print(f"  Layer {i}: {feat.shape}")

    # 创建decoder
    decoder = Mask2FormerDecoder(
        feature_dim=feature_dim,
        hidden_dim=256,
        num_classes=num_classes,
        num_queries=100,
        num_decoder_layers=6
    )

    # 前向传播
    print("\nForward pass...")
    outputs = decoder(multi_scale_features, image_size)

    print(f"\nOutputs:")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_masks: {outputs['pred_masks'].shape}")
    print(f"  num aux outputs: {len(outputs['aux_outputs'])}")

    # 转换为语义分割
    semantic_logits = decoder.get_semantic_segmentation(outputs, image_size)
    print(f"\nSemantic logits: {semantic_logits.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("✅ Test passed!")
    print("=" * 70)
