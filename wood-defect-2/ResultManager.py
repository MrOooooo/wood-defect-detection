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

from pandas.io.formats.printing import justify

from GUI.Data.BatchResult import BatchResult
from GUI.Data.InterfenceResult import InferenceResult
from GUI.Tool.ImageProcessor import ImageProcessor


class ResultsManager:
    """结果管理类"""

    def __init__(self, vis_frame: tk.Frame, results_tree: ttk.Treeview,
                 available_models: Dict, dataset_type: str):
        self.vis_frame = vis_frame
        self.results_tree = results_tree
        self.available_models = available_models
        self.dataset_type = dataset_type
        self.image_processor = ImageProcessor()

    def display_single_result(self, image_path: str, results: Dict[str, InferenceResult]):
        """显示单图结果"""
        # 清空表格
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # 填充表格
        for model_name, result in results.items():
            display_name = self.available_models[model_name]['name']
            self.results_tree.insert('', 'end', values=(
                display_name,
                f"{result.metrics['mIoU']:.4f}",
                f"{result.metrics['mAcc']:.4f}",
                f"{result.metrics['F1']:.4f}",
                f"{result.inference_time:.2f}"
            ))

        # 清空可视化区域
        for widget in self.vis_frame.winfo_children():
            widget.destroy()

        # 创建网格布局
        row, col = 0, 0
        max_cols = 4

        # 添加原始图片
        self._add_result_image(image_path, '原始图片', row, col)
        col += 1

        # 添加真实标签(如果存在)
        label_path = self.image_processor.get_label_path(image_path)
        if label_path and os.path.exists(label_path):
            label_image = Image.open(label_path)
            label_array = np.array(label_image)
            label_colored = self.image_processor.colorize_segmentation(label_array, self.dataset_type)

            original_image = Image.open(image_path).convert('RGB')
            original_image = original_image.resize(
                (label_colored.shape[1], label_colored.shape[0]),
                Image.Resampling.LANCZOS
            )
            original_array = np.array(original_image)
            label_overlay = self.image_processor.create_overlay(original_array, label_colored)

            self._add_result_image_from_array(
                label_overlay,
                '真实标签 (Ground Truth)',
                row, col
            )
            col += 1

        # 添加各模型结果
        for model_name, result in results.items():
            if col >= max_cols:
                col = 0
                row += 1

            display_name = self.available_models[model_name]['name']
            self._add_result_image_from_array(
                result.segmentation,
                f"{display_name}\nmIoU: {result.metrics['mIoU']:.4f}",
                row, col
            )
            col += 1

    def display_batch_results(self, batch_results: List[BatchResult]):
        """显示批量结果 - 按照单图格式显示每张图片"""
        # 清空表格
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(batch_results)

        # 填充表格 - 显示平均值
        for model_name, metrics in avg_metrics.items():
            display_name = self.available_models[model_name]['name']
            self.results_tree.insert('', 'end', values=(
                f"{display_name} (平均)",
                f"{metrics['mIoU']:.4f}",
                f"{metrics['mAcc']:.4f}",
                f"{metrics['F1']:.4f}",
                f"{metrics['avg_time']:.2f}"
            ))

        # 清空可视化区域
        for widget in self.vis_frame.winfo_children():
            widget.destroy()

        for batch_idx, batch_result in enumerate(batch_results):
            # 创建一个容器框架来包含每张图片的所有内容
            image_container = tk.Frame(self.vis_frame, bg='white')
            image_container.grid(row=batch_idx, column=0, sticky='ew', padx=20, pady=20)

            # 分隔线标题（放在容器内）
            separator = tk.Label(
                image_container,
                text=f"图片 {batch_idx + 1}/{len(batch_results)}: {os.path.basename(batch_result.image_path)} | 平均处理时间: {batch_result.avg_time:.2f}ms",
                font=('newspaper', 12, 'bold'),
                bg='white',
                fg='#2563eb',
                anchor='w'
            )
            separator.pack(fill='x', pady=(0, 10))

            # 分隔线
            tk.Frame(image_container, height=2, bg='#e5e7eb').pack(fill='x', pady=(0, 15))

            # 图片网格容器
            images_grid = tk.Frame(image_container, bg='white')
            images_grid.pack(fill='both', expand=True)

            col = 0
            max_cols = 4

            # 1. 显示原始图片
            self._add_result_image_to_frame(
                images_grid,
                batch_result.image_path,
                '原始图片',
                0, col
            )
            col += 1

            # 2. 显示真实标签（如果存在）
            label_path = self.image_processor.get_label_path(batch_result.image_path)
            if label_path and os.path.exists(label_path):
                label_image = Image.open(label_path)
                label_array = np.array(label_image)
                label_colored = self.image_processor.colorize_segmentation(
                    label_array,
                    self.dataset_type
                )

                original_image = Image.open(batch_result.image_path).convert('RGB')
                original_image = original_image.resize(
                    (label_colored.shape[1], label_colored.shape[0]),
                    Image.Resampling.LANCZOS
                )
                original_array = np.array(original_image)
                label_overlay = self.image_processor.create_overlay(original_array, label_colored)

                self._add_result_image_from_array_to_frame(
                    images_grid,
                    label_overlay,
                    '真实标签 (Ground Truth)',
                    0, col
                )
                col += 1

            # 3. 显示各个模型的结果
            row = 0
            for model_name, result in batch_result.results.items():
                if col >= max_cols:
                    col = 0
                    row += 1

                display_name = self.available_models[model_name]['name']
                self._add_result_image_from_array_to_frame(
                    images_grid,
                    result.segmentation,
                    f"{display_name}\nmIoU: {result.metrics['mIoU']:.4f}\n时间: {result.inference_time:.2f}ms",
                    row, col
                )
                col += 1

    def _calculate_average_metrics(self, batch_results: List[BatchResult]) -> Dict:
        """计算批量结果的平均指标"""
        model_metrics = {}

        for batch_result in batch_results:
            for model_name, result in batch_result.results.items():
                if model_name not in model_metrics:
                    model_metrics[model_name] = {
                        'mIoU': [],
                        'mAcc': [],
                        'F1': [],
                        'time': []
                    }

                model_metrics[model_name]['mIoU'].append(result.metrics['mIoU'])
                model_metrics[model_name]['mAcc'].append(result.metrics['mAcc'])
                model_metrics[model_name]['F1'].append(result.metrics['F1'])
                model_metrics[model_name]['time'].append(result.inference_time)

        avg_metrics = {}
        for model_name, metrics in model_metrics.items():
            avg_metrics[model_name] = {
                'mIoU': np.mean(metrics['mIoU']),
                'mAcc': np.mean(metrics['mAcc']),
                'F1': np.mean(metrics['F1']),
                'avg_time': np.mean(metrics['time'])
            }

        return avg_metrics

    def _add_result_image(self, image_path: str, title: str, row: int, col: int):
        """添加结果图片"""
        frame = tk.Frame(self.vis_frame, bg='white', relief='solid', borderwidth=1)
        frame.grid(row=row, column=col, padx=10, pady=10)

        title_label = tk.Label(
            frame,
            text=title,
            font=('newspaper', 11, 'bold'),
            bg='white'
        )
        title_label.pack(pady=5)

        image = Image.open(image_path)
        image.thumbnail((350, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        img_label = tk.Label(frame, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(padx=5, pady=5)

    def _add_result_image_from_array(self, image: np.ndarray, title: str, row: int, col: int):
        """从数组添加结果图片"""
        frame = tk.Frame(self.vis_frame, bg='white', relief='solid', borderwidth=1)
        frame.grid(row=row, column=col, padx=10, pady=10)

        title_label = tk.Label(
            frame,
            text=title,
            font=('newspaper', 11, 'bold'),
            bg='white'
        )
        title_label.pack(pady=5)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image.thumbnail((250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        img_label = tk.Label(frame, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(padx=5, pady=5)

    def _add_result_image_to_frame(self, parent_frame: tk.Frame, image_path: str,
                                   title: str, row: int, col: int):
        """添加结果图片到指定框架"""
        frame = tk.Frame(parent_frame, bg='white', relief='solid', borderwidth=1)
        frame.grid(row=row, column=col, padx=10, pady=10)

        title_label = tk.Label(
            frame,
            text=title,
            font=('newspaper', 11, 'bold'),
            bg='white'
        )
        title_label.pack(pady=5)

        image = Image.open(image_path)
        image.thumbnail((250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        img_label = tk.Label(frame, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(padx=5, pady=5)

    def _add_result_image_from_array_to_frame(self, parent_frame: tk.Frame,
                                              image: np.ndarray, title: str,
                                              row: int, col: int):
        """从数组添加结果图片到指定框架"""
        frame = tk.Frame(parent_frame, bg='white', relief='solid', borderwidth=1)
        frame.grid(row=row, column=col, padx=10, pady=10)

        title_label = tk.Label(
            frame,
            text=title,
            font=('newspaper', 11, 'bold'),
            bg='white'
        )
        title_label.pack(pady=5)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image.thumbnail((250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        img_label = tk.Label(frame, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(padx=5, pady=5)

