import os

import torch
from PIL import Image, ImageTk
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class ModelLoader:
    """模型加载工具类"""

    def __init__(self, checkpoint_dir: str, device: str = 'cuda:1'):
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def load_model(self, model_name: str, num_classes: int):
        """加载指定模型"""
        if model_name == 'lam':
            return self._load_lam_model(num_classes)
        else:
            return self._load_benchmark_model(model_name, num_classes)

    def _load_lam_model(self, num_classes: int):
        """加载LAM模型"""
        from models import LAMSegmentationModel

        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            has_lsm = any('lsm' in key for key in state_dict.keys())

            model = LAMSegmentationModel(
                backbone_name='dinov2',
                num_classes=num_classes,
                num_tokens=100,
                token_rank=16,
                num_groups=16,
                use_lsm=has_lsm
            ).to(self.device)

            try:
                model.load_state_dict(state_dict)
                print(f"✓ 成功加载LAM模型")
            except Exception as e:
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ 使用非严格模式加载模型")
        else:
            print(f"警告: 未找到模型文件 {checkpoint_path}")
            model = LAMSegmentationModel(
                backbone_name='dinov2',
                num_classes=num_classes,
                num_tokens=100,
                token_rank=16,
                num_groups=16,
                use_lsm=False
            ).to(self.device)

        return model

    def _load_benchmark_model(self, model_name: str, num_classes: int):
        """加载基准模型"""
        from benchmark_comparison import ModelFactory

        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False
        ).to(self.device)

        checkpoint_path = os.path.join(
            os.path.dirname(self.checkpoint_dir),
            f'benchmark_{model_name}',
            'best_model.pth'
        )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ 成功加载 {model_name} 模型")
            except Exception as e:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print(f"警告: 未找到模型文件 {checkpoint_path}")

        return model