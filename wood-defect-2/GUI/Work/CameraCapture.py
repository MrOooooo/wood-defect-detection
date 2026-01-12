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


class CameraCapture:
    """摄像头采集类"""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)

        self.frame_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None

    def set_callbacks(self, frame_cb: Callable, error_cb: Callable):
        """设置回调函数"""
        self.frame_callback = frame_cb
        self.error_callback = error_cb

    def start(self) -> bool:
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")

            self.is_running = True
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            thread.start()
            return True

        except Exception as e:
            if self.error_callback:
                self.error_callback(f"摄像头启动失败: {str(e)}")
            return False

    def stop(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        """采集循环"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 清空队列并放入新帧
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

                try:
                    self.frame_queue.put_nowait(frame_rgb)
                except queue.Full:
                    pass

                if self.frame_callback:
                    self.frame_callback(frame_rgb)

            time.sleep(0.03)  # ~30fps

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """捕获快照"""
        return self.get_frame()
