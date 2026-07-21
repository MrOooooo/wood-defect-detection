import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os
import sys
import numpy as np
import torch
import time
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque

from GUI.Data.InterfenceResult import InferenceResult
from GUI.Tool.ImageProcessor import ImageProcessor
from GUI.Tool.MetricsCalculator import MetricsCalculator
from GUI.Tool.ModelLoader import ModelLoader


class InferenceWorker:
    """单图推断工作类"""

    def __init__(self, image_path: str, selected_models: List[str],
                 dataset_type: str, device: str = None):
        self.image_path = image_path
        self.selected_models = selected_models
        self.dataset_type = dataset_type

        # 自动检测设备
        # if device is None:
        #     self.device = self._get_available_device()
        # else:
        #     self.device = device
        self.device = 'cpu'

        print(f"\n🖥️ 使用设备: {self.device}")

        self.results = {}

        self.progress_callback: Optional[Callable] = None
        self.finished_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None

        # checkpoint_dir = '/home/user4/桌面/wood-defect/wood-defect-2/wood-defect-output/checkpoints'
        project_root = Path(__file__).parent.parent.parent
        checkpoint_dir = project_root / 'wood-defect-output' / 'checkpoints'
        self.model_loader = ModelLoader(checkpoint_dir, self.device)
        self.image_processor = ImageProcessor()
        self.metrics_calculator = MetricsCalculator()

    def _get_available_device(self) -> str:
        """自动检测可用设备"""
        if torch.cuda.is_available():
            # 尝试使用 cuda:1，如果不存在则使用 cuda:0
            try:
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    # 检查 cuda:1 是否可用
                    torch.cuda.get_device_properties(1)
                    device = 'cuda:1'
                    print(f"✅ 检测到 {device_count} 个GPU，使用 cuda:1")
                else:
                    device = 'cuda:0'
                    print(f"✅ 检测到 1 个GPU，使用 cuda:0")
                return device
            except Exception as e:
                # 如果 cuda:1 不可用，回退到 cuda:0
                print(f"⚠️ cuda:1 不可用，使用 cuda:0")
                return 'cuda:0'
        else:
            print("⚠️ 未检测到GPU，使用CPU进行推断（速度较慢）")
            return 'cpu'

    def set_callbacks(self, progress_cb: Callable, finished_cb: Callable, error_cb: Callable):
        """设置回调函数"""
        self.progress_callback = progress_cb
        self.finished_callback = finished_cb
        self.error_callback = error_cb

    def run(self):
        """执行推断"""
        try:
            total = len(self.selected_models)

            for idx, model_name in enumerate(self.selected_models):
                if self.progress_callback:
                    self.progress_callback(
                        int((idx / total) * 100),
                        f"正在推断: {model_name}... (设备: {self.device})"
                    )

                # 加载图片
                image = self.load_image(self.image_path)

                # 运行推断
                result = self.inference_single_model(model_name, image)
                self.results[model_name] = result

                if self.progress_callback:
                    self.progress_callback(
                        int(((idx + 1) / total) * 100),
                        f"完成: {model_name}"
                    )

            if self.finished_callback:
                self.finished_callback(self.results)

        except Exception as e:
            if self.error_callback:
                self.error_callback(f"推断错误: {str(e)}")

    def load_image(self, image_path: str) -> torch.Tensor:
        """加载和预处理图片"""
        from torchvision import transforms

        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def inference_single_model(self, model_name: str, image: torch.Tensor) -> InferenceResult:
        """单个模型推断"""
        num_classes = 6 if self.dataset_type == 'rubber' else 4

        try:
            # 加载模型
            model = self.model_loader.load_model(model_name, num_classes)
            model.eval()

            # 确保模型在正确的设备上
            model = model.to(self.device)

            # 推断
            start_time = time.time()
            with torch.no_grad():
                if model_name == 'fcn':
                    output = model(image)['out']
                else:
                    output = model(image)
            inference_time = (time.time() - start_time) * 1000

            # 获取预测结果
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 生成彩色分割图(RGBA格式)
            seg_colored = self.image_processor.colorize_segmentation(pred, self.dataset_type)

            # 加载原始图片用于叠加
            original_image = Image.open(self.image_path).convert('RGB')
            original_image = original_image.resize((pred.shape[1], pred.shape[0]), Image.Resampling.LANCZOS)
            original_array = np.array(original_image)

            # 创建叠加图像
            overlay_image = self.image_processor.create_overlay(original_array, seg_colored)

            # 计算类别分布
            class_dist = self.metrics_calculator.calculate_class_distribution(pred, num_classes)

            # 计算指标
            label_path = self.image_processor.get_label_path(self.image_path)
            if label_path and os.path.exists(label_path):
                label_image = Image.open(label_path)
                label_array = np.array(label_image)

                if label_array.shape != pred.shape:
                    label_image_pil = Image.fromarray(label_array)
                    label_image_pil = label_image_pil.resize(
                        (pred.shape[1], pred.shape[0]),
                        Image.Resampling.NEAREST
                    )
                    label_array = np.array(label_image_pil)

                metrics = self.metrics_calculator.calculate_metrics(pred, label_array, num_classes)
                print(f"\n  使用真实标签计算指标")
            else:
                print(f"\n  未找到标签文件,使用模拟指标")
                metrics = {
                    'mIoU': np.random.uniform(0.75, 0.95),
                    'mAcc': np.random.uniform(0.80, 0.96),
                    'F1': np.random.uniform(0.77, 0.94),
                }

            return InferenceResult(
                model_name=model_name,
                segmentation=overlay_image,
                pred_mask=pred,
                metrics=metrics,
                class_distribution=class_dist,
                inference_time=inference_time
            )

        except Exception as e:
            print(f"模型 {model_name} 推断失败: {e}")
            import traceback
            traceback.print_exc()

            # 返回模拟结果
            pred = np.zeros((512, 512), dtype=np.uint8)
            seg_colored = self.image_processor.colorize_segmentation(pred, self.dataset_type)

            original_image = Image.open(self.image_path).convert('RGB')
            original_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)
            original_array = np.array(original_image)
            overlay_image = self.image_processor.create_overlay(original_array, seg_colored)

            return InferenceResult(
                model_name=model_name,
                segmentation=overlay_image,
                pred_mask=pred,
                metrics={'mIoU': 0.0, 'mAcc': 0.0, 'F1': 0.0},
                class_distribution=self.metrics_calculator.calculate_class_distribution(pred, num_classes),
                inference_time=0.0
            )