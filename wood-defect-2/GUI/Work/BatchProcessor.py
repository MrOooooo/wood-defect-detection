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

from GUI.Data.BatchResult import BatchResult
from GUI.Work.InferenceWorker import InferenceWorker


class BatchProcessor:
    """批量图片处理类"""

    def __init__(self, image_paths: List[str], selected_models: List[str],
                 dataset_type: str, device: str = 'cuda:1'):
        self.image_paths = image_paths
        self.selected_models = selected_models
        self.dataset_type = dataset_type
        self.device = device
        self.batch_results: List[BatchResult] = []

        self.progress_callback: Optional[Callable] = None
        self.item_finished_callback: Optional[Callable] = None
        self.finished_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None

    def set_callbacks(self, progress_cb: Callable, item_finished_cb: Callable,
                      finished_cb: Callable, error_cb: Callable):
        """设置回调函数"""
        self.progress_callback = progress_cb
        self.item_finished_callback = item_finished_cb
        self.finished_callback = finished_cb
        self.error_callback = error_cb

    def run(self):
        """执行批量处理"""
        try:
            total = len(self.image_paths)

            for idx, image_path in enumerate(self.image_paths):
                if self.progress_callback:
                    self.progress_callback(
                        int((idx / total) * 100),
                        f"处理 {idx + 1}/{total}: {os.path.basename(image_path)}"
                    )

                # 为每张图片创建推断工作器
                worker = InferenceWorker(
                    image_path,
                    self.selected_models,
                    self.dataset_type,
                    self.device
                )

                # 同步运行
                worker.run()

                # 计算平均时间
                avg_time = np.mean([r.inference_time for r in worker.results.values()])

                batch_result = BatchResult(
                    image_path=image_path,
                    results=worker.results,
                    avg_time=avg_time
                )

                self.batch_results.append(batch_result)

                # 单项完成回调
                if self.item_finished_callback:
                    self.item_finished_callback(batch_result)

            if self.finished_callback:
                self.finished_callback(self.batch_results)

        except Exception as e:
            if self.error_callback:
                self.error_callback(f"批量处理错误: {str(e)}")
