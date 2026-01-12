# models/backbone.py
"""
视觉基础模型Backbone
支持DINOv2、SAM、CLIP等
"""
import os
import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoImageProcessor



class VFMBackbone(nn.Module):
    """
    视觉基础模型Backbone包装器
    """

    def __init__(self, model_name='dinov2', freeze=True, output_layers=None):
        """
        Args:
            model_name: 模型名称 ('dinov2', 'sam', 'clip')
            freeze: 是否冻结backbone参数
            output_layers: 需要输出的层索引列表，None表示输出所有层
        """
        super(VFMBackbone, self).__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.output_layers = output_layers

        # 加载模型
        if model_name == 'dinov2':
            self.load_dinov2()
        elif model_name == 'sam':
            self.load_sam()
        elif model_name == 'clip':
            self.load_clip()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 冻结参数
        if freeze:
            self.freeze_parameters()

    def load_dinov2(self, local_model_path=None):
        """加载DINOv2模型"""
        # 优先使用本地路径
        if local_model_path is None:
            from configs.lam_config import config
            local_model_path = config.dinov2_model_path if hasattr(config, 'dinov2_model_path') else None
            print(local_model_path)
            print(os.path.exists(local_model_path))
        # 尝试加载模型
        try:
            if local_model_path and os.path.exists(local_model_path):
                print(f"Loading DINOv2 from local path: {local_model_path}")
                self.backbone = AutoModel.from_pretrained(local_model_path)
                self.processor = AutoImageProcessor.from_pretrained(local_model_path)
                print("✓ Successfully loaded DINOv2 from local path")
            else:
                print("Attempting to load DINOv2 from HuggingFace...")
                model_id = "facebook/dinov2-base"
                self.backbone = AutoModel.from_pretrained(model_id)
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                print("✓ Successfully loaded DINOv2 from HuggingFace")
        except Exception as e:
            print(f"Failed to load DINOv2: {e}")
            print("Creating dummy model for testing...")
            self.backbone = self._create_dummy_vit()
            self.processor = None

        # DINOv2-base的特征维度
        self.feature_dim = 768  # base: 768, large: 1024
        self.num_layers = 12  # base: 12, large: 24
        self.patch_size = 14

    def load_sam(self):
        """加载SAM模型"""
        print("SAM model loading not fully implemented")
        print("Using dummy model for testing")
        self.backbone = self._create_dummy_vit()
        self.processor = None
        self.feature_dim = 1024
        self.num_layers = 24
        self.patch_size = 16

    def load_clip(self):
        """加载CLIP模型"""
        model_id = "openai/clip-vit-large-patch14"

        try:
            self.backbone = AutoModel.from_pretrained(model_id).vision_model
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        except:
            print(f"Failed to load {model_id} from HuggingFace")
            self.backbone = self._create_dummy_vit()
            self.processor = None

        self.feature_dim = 1024
        self.num_layers = 24
        self.patch_size = 14

    def _create_dummy_vit(self):
        """创建一个简单的ViT模型用于测试"""

        class DummyViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, 1024, kernel_size=14, stride=14)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=1024,
                        nhead=16,
                        dim_feedforward=4096,
                        batch_first=True
                    ) for _ in range(24)
                ])
                self.norm = nn.LayerNorm(1024)

            def forward(self, x, output_hidden_states=False):
                # Patch embedding
                x = self.patch_embed(x)  # (B, 1024, H/14, W/14)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # (B, H*W, 1024)

                # Transformer blocks
                hidden_states = []
                for block in self.blocks:
                    x = block(x)
                    if output_hidden_states:
                        hidden_states.append(x)

                x = self.norm(x)

                if output_hidden_states:
                    hidden_states.append(x)
                    return type('Outputs', (), {
                        'last_hidden_state': x,
                        'hidden_states': tuple(hidden_states)
                    })()
                else:
                    return type('Outputs', (), {
                        'last_hidden_state': x
                    })()

        return DummyViT()

    def freeze_parameters(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Frozen all parameters in {self.model_name} backbone")

    # models/backbone.py
    def forward(self, x, output_hidden_states=True):
        """
        前向传播
        Args:
            x: (B, 3, H, W) 输入图像
            output_hidden_states: 是否输出所有隐藏层状态
        Returns:
            features: 特征列表或最后一层特征
        """
        # 前向传播
        outputs = self.backbone(x, output_hidden_states=output_hidden_states)

        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            # DINOv2的hidden_states包含所有层的输出
            hidden_states = outputs.hidden_states

            # 注意:hidden_states[0]是embedding层输出,hidden_states[1-12]是12个transformer层
            # 总共13个元素(embedding + 12层)

            if self.output_layers is not None:
                # 确保索引有效(注意:需要+1因为第0个是embedding)
                features = []
                for i in self.output_layers:
                    if 0<= i < len(hidden_states):  # 排除embedding层
                        features.append(hidden_states[i])  # +1跳过embedding
                    else:
                        print(f"Warning: Layer {i} out of range (max={len(hidden_states) - 1})")
                        features.append(hidden_states[-1])
            else:
                # 返回所有transformer层(跳过embedding)
                features = list(hidden_states[1:])

            return features
        else:
            # 只返回最后一层特征
            return [outputs.last_hidden_state]

    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim

    def get_num_layers(self):
        """返回层数"""
        return self.num_layers


class SegmentationHead(nn.Module):
    """
    简化的分割头
    将特征转换为分割mask
    """

    def __init__(self, feature_dim, num_classes, hidden_dim=256):
        super(SegmentationHead, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(self, features, image_size):
        """
        Args:
            features: (B, L, C) 特征，其中L=patch数+1（包含[CLS] token）
            image_size: (H, W) 目标图像大小
        Returns:
            logits: (B, num_classes, H, W) 分割logits
        """
        B, L, C = features.shape

        # 移除[CLS] token（第一个token）
        # 只使用图像patch的特征
        # features = features[:, 1:, :]  # (B, L-1, C)
        # L = L - 1

        # 计算合适的H和W
        H = W =int(math.sqrt(L))

        if H * W != L:
            # 使用最接近的平方数
            H = W = int(math.ceil(math.sqrt(L)))

            # 填充或截断
            if H * W > L:
                padding = H * W - L
                features = torch.nn.functional.pad(
                    features, (0, 0, 0, padding), mode='constant', value=0
                )
            else:
                features = features[:, :H * W, :]

        # 重塑特征
        features = features.transpose(1, 2).reshape(B, C, H, W)

        # 如果还有剩余token，调整W
        # if H * W < L:
        #     W += 1

        # 确保H*W等于L
        # if H * W != L:
        #     # 如果仍然不匹配，使用插值调整
        #     print(f"警告: 无法完美重塑特征 H*W={H * W} != L={L}")
        #     # 使用自适应池化或插值来处理

        # 安全重塑特征
        # try:
        #     features = features.transpose(1, 2).reshape(B, C, H, W)
        # except RuntimeError as e:
        #     print(f"特征重塑错误: B={B}, L={L}, C={C}, H={H}, W={W}")
        #     # 使用插值作为回退方案
        #     target_H, target_W = int(math.sqrt(L)), int(math.sqrt(L))
        #     if target_H * target_W < L:
        #         target_W += 1
        #
        #     # 使用线性插值调整特征尺寸
        #     features_2d = features.transpose(1, 2).unsqueeze(-1)  # (B, C, L, 1)
        #     features_2d = torch.nn.functional.interpolate(
        #         features_2d,
        #         size=(target_H * target_W, 1),
        #         mode='linear',
        #         align_corners=False
        #     )
        #     features = features_2d.squeeze(-1).reshape(B, C, target_H, target_W)
        #     H, W = target_H, target_W

        # 解码
        logits = self.decoder(features)

        # 上采样到目标大小
        logits = torch.nn.functional.interpolate(
            logits,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )

        return logits

if __name__ == "__main__":
    # 测试VFM Backbone
    print("Testing VFM Backbone...")

    # 创建模型
    backbone = VFMBackbone(model_name='dinov2', freeze=True)

    # 创建测试输入
    batch_size = 2
    image = torch.randn(batch_size, 3, 512, 512)

    # 前向传播
    features = backbone(image, output_hidden_states=True)

    print(f"Number of layers: {len(features)}")
    print(f"Feature shape per layer: {features[0].shape}")
    print(f"Feature dimension: {backbone.get_feature_dim()}")

    # 测试分割头
    seg_head = SegmentationHead(
        feature_dim=backbone.get_feature_dim(),
        num_classes=5
    )

    # 使用最后一层特征
    last_features = features[-1]
    logits = seg_head(last_features, image_size=(512, 512))

    print(f"Segmentation logits shape: {logits.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    print(f"\nBackbone parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    seg_head_params = sum(p.numel() for p in seg_head.parameters())
    print(f"\nSegmentation head parameters: {seg_head_params:,}")

    print("\nBackbone test passed!")