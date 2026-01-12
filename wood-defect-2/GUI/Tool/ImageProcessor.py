import os
from PIL import Image, ImageTk
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def colorize_segmentation(pred_mask: np.ndarray, dataset_type: str) -> np.ndarray:
        """将分割mask转换为RGBA彩色图"""
        color_maps = {
            'rubber': [
                [0, 0, 0, 0],  # Background (完全透明)
                [255, 0, 0, 150],  # Dead Knot
                [0, 255, 0, 150],  # Sound Knot
                [0, 100, 255, 150],  # Missing Edge
                [255, 255, 0, 150],  # Timber Core
                [255, 0, 255, 150],  # Crack
            ],
            'pine': [
                [0, 0, 0, 0],
                [255, 0, 0, 150],
                [0, 255, 0, 150],
                [0, 100, 255, 150],
            ]
        }

        colors = color_maps[dataset_type]
        h, w = pred_mask.shape
        colored = np.zeros((h, w, 4), dtype=np.uint8)

        for class_id, color in enumerate(colors):
            mask = pred_mask == class_id
            colored[mask] = color

        return colored

    @staticmethod
    def create_overlay(original_rgb: np.ndarray, segmentation_rgba: np.ndarray) -> np.ndarray:
        """将分割结果叠加到原图上"""
        # 确保尺寸匹配
        if original_rgb.shape[:2] != segmentation_rgba.shape[:2]:
            seg_img = Image.fromarray(segmentation_rgba)
            seg_img = seg_img.resize(
                (original_rgb.shape[1], original_rgb.shape[0]),
                Image.Resampling.NEAREST
            )
            segmentation_rgba = np.array(seg_img)

        overlay = original_rgb.copy().astype(np.float32)
        seg_rgb = segmentation_rgba[:, :, :3].astype(np.float32)
        seg_alpha = segmentation_rgba[:, :, 3].astype(np.float32) / 255.0

        for c in range(3):
            overlay[:, :, c] = (
                    overlay[:, :, c] * (1 - seg_alpha) +
                    seg_rgb[:, :, c] * seg_alpha
            )

        return overlay.astype(np.uint8)

    @staticmethod
    def get_label_path(image_path: str) -> Optional[str]:
        """根据图片路径获取对应的标签路径"""
        try:
            parent_dir = os.path.dirname(os.path.dirname(image_path))
            filename = os.path.splitext(os.path.basename(image_path))[0]

            possible_paths = [
                os.path.join(parent_dir, 'SegmentationClass', f'{filename}.png'),
                os.path.join(parent_dir, 'labels', f'{filename}.png'),
                os.path.join(parent_dir, 'masks', f'{filename}.png'),
                os.path.join(os.path.dirname(image_path), 'labels', f'{filename}.png'),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return None
        except:
            return None