import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
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
from datetime import datetime



# 导入原有模块
from GUI.Data.BatchResult import BatchResult
from GUI.Data.InterfenceResult import InferenceResult
from GUI.Tool.ResultManager import ResultsManager
from GUI.Work.BatchProcessor import BatchProcessor
from GUI.Work.CameraCapture import CameraCapture
from GUI.Work.IPCameraCaputer import IPCameraCapture
from GUI.Work.InferenceWorker import InferenceWorker
from GUI.Work.VideoCapture import VideoCapture


class IPCameraConfigDialog:
    """IP 摄像头配置对话框"""

    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title('IP 摄像头配置')
        self.dialog.geometry('600x500')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._create_widgets()

    def _create_widgets(self):
        """创建控件"""
        main_frame = tk.Frame(self.dialog, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        # 标题
        title = tk.Label(
            main_frame,
            text='配置 IP 摄像头',
            font=('微软雅黑', 16, 'bold'),
            fg='#2563eb'
        )
        title.pack(pady=(0, 20))

        # 配置方式选择
        mode_frame = tk.LabelFrame(main_frame, text='配置方式', font=('微软雅黑', 11, 'bold'), padx=10, pady=10)
        mode_frame.pack(fill='x', pady=10)

        self.mode_var = tk.StringVar(value='template')

        tk.Radiobutton(
            mode_frame,
            text='使用模板配置',
            variable=self.mode_var,
            value='template',
            command=self._toggle_mode,
            font=('微软雅黑', 10)
        ).pack(anchor='w', pady=2)

        tk.Radiobutton(
            mode_frame,
            text='直接输入 URL',
            variable=self.mode_var,
            value='url',
            command=self._toggle_mode,
            font=('微软雅黑', 10)
        ).pack(anchor='w', pady=2)

        # 模板配置区域
        self.template_frame = tk.LabelFrame(
            main_frame,
            text='模板配置',
            font=('微软雅黑', 11, 'bold'),
            padx=10,
            pady=10
        )
        self.template_frame.pack(fill='x', pady=10)

        # 摄像头类型
        tk.Label(self.template_frame, text='摄像头类型:', font=('微软雅黑', 10)).grid(row=0, column=0, sticky='w',
                                                                                      pady=5)
        self.type_var = tk.StringVar(value='hikvision_rtsp')
        type_combo = ttk.Combobox(
            self.template_frame,
            textvariable=self.type_var,
            values=['hikvision_rtsp', 'dahua_rtsp', 'generic_rtsp', 'http_mjpeg', 'onvif'],
            state='readonly',
            width=30
        )
        type_combo.grid(row=0, column=1, pady=5, padx=(10, 0))

        # IP 地址
        tk.Label(self.template_frame, text='IP 地址:', font=('微软雅黑', 10)).grid(row=1, column=0, sticky='w', pady=5)
        self.ip_entry = tk.Entry(self.template_frame, width=32, font=('微软雅黑', 10))
        self.ip_entry.insert(0, '192.168.1.64')
        self.ip_entry.grid(row=1, column=1, pady=5, padx=(10, 0))

        # 端口
        tk.Label(self.template_frame, text='端口:', font=('微软雅黑', 10)).grid(row=2, column=0, sticky='w', pady=5)
        self.port_entry = tk.Entry(self.template_frame, width=32, font=('微软雅黑', 10))
        self.port_entry.insert(0, '554')
        self.port_entry.grid(row=2, column=1, pady=5, padx=(10, 0))

        # 用户名
        tk.Label(self.template_frame, text='用户名:', font=('微软雅黑', 10)).grid(row=3, column=0, sticky='w', pady=5)
        self.user_entry = tk.Entry(self.template_frame, width=32, font=('微软雅黑', 10))
        self.user_entry.insert(0, 'admin')
        self.user_entry.grid(row=3, column=1, pady=5, padx=(10, 0))

        # 密码
        tk.Label(self.template_frame, text='密码:', font=('微软雅黑', 10)).grid(row=4, column=0, sticky='w', pady=5)
        self.password_entry = tk.Entry(self.template_frame, width=32, show='*', font=('微软雅黑', 10))
        self.password_entry.insert(0, 'admin')
        self.password_entry.grid(row=4, column=1, pady=5, padx=(10, 0))

        # 通道号
        tk.Label(self.template_frame, text='通道号:', font=('微软雅黑', 10)).grid(row=5, column=0, sticky='w', pady=5)
        self.channel_entry = tk.Entry(self.template_frame, width=32, font=('微软雅黑', 10))
        self.channel_entry.insert(0, '1')
        self.channel_entry.grid(row=5, column=1, pady=5, padx=(10, 0))

        # URL 配置区域
        self.url_frame = tk.LabelFrame(
            main_frame,
            text='URL 配置',
            font=('微软雅黑', 11, 'bold'),
            padx=10,
            pady=10
        )

        tk.Label(
            self.url_frame,
            text='完整 RTSP/HTTP URL:',
            font=('微软雅黑', 10)
        ).pack(anchor='w', pady=(0, 5))

        self.url_entry = tk.Entry(self.url_frame, width=50, font=('微软雅黑', 10))
        self.url_entry.insert(0, 'rtsp://admin:password@192.168.1.64:554/stream1')
        self.url_entry.pack(fill='x', pady=5)

        # 示例说明
        example_text = """
常见格式示例:
• 海康威视: rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
• 大华: rtsp://admin:password@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0
• 通用 RTSP: rtsp://admin:password@192.168.1.64:554/stream1
• HTTP MJPEG: http://192.168.1.64:8080/video
        """
        tk.Label(
            self.url_frame,
            text=example_text,
            font=('微软雅黑', 8),
            fg='#6b7280',
            justify='left'
        ).pack(anchor='w', pady=(10, 0))

        # 按钮
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame,
            text='测试连接',
            command=self._test_connection,
            bg='#f59e0b',
            fg='white',
            font=('微软雅黑', 10, 'bold'),
            padx=20,
            pady=5
        ).pack(side='left', padx=5)

        tk.Button(
            btn_frame,
            text='确定',
            command=self._on_ok,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 10, 'bold'),
            padx=20,
            pady=5
        ).pack(side='left', padx=5)

        tk.Button(
            btn_frame,
            text='取消',
            command=self._on_cancel,
            bg='#6b7280',
            fg='white',
            font=('微软雅黑', 10, 'bold'),
            padx=20,
            pady=5
        ).pack(side='left', padx=5)

        self._toggle_mode()

    def _toggle_mode(self):
        """切换配置模式"""
        if self.mode_var.get() == 'template':
            self.template_frame.pack(fill='x', pady=10)
            self.url_frame.pack_forget()
        else:
            self.template_frame.pack_forget()
            self.url_frame.pack(fill='x', pady=10)

    def _get_config(self) -> Dict:
        """获取配置"""
        if self.mode_var.get() == 'template':
            return {
                'mode': 'template',
                'camera_type': self.type_var.get(),
                'ip': self.ip_entry.get(),
                'port': int(self.port_entry.get()),
                'user': self.user_entry.get(),
                'password': self.password_entry.get(),
                'channel': int(self.channel_entry.get()),
            }
        else:
            return {
                'mode': 'url',
                'camera_url': self.url_entry.get()
            }

    def _test_connection(self):
        """测试连接"""
        config = self._get_config()

        # 显示测试中对话框
        test_dialog = tk.Toplevel(self.dialog)
        test_dialog.title('测试连接')
        test_dialog.geometry('400x200')
        test_dialog.transient(self.dialog)
        test_dialog.grab_set()

        tk.Label(
            test_dialog,
            text='正在测试连接...',
            font=('微软雅黑', 12)
        ).pack(pady=50)

        result_label = tk.Label(
            test_dialog,
            text='',
            font=('微软雅黑', 10)
        )
        result_label.pack()

        def test():
            try:
                if config['mode'] == 'template':
                    camera = IPCameraCapture(
                        camera_type=config['camera_type'],
                        ip=config['ip'],
                        port=config['port'],
                        user=config['user'],
                        password=config['password'],
                        channel=config['channel']
                    )
                else:
                    camera = IPCameraCapture(camera_url=config['camera_url'])

                success = camera.test_connection()

                if success:
                    result_label.config(text='✅ 连接成功!', fg='green')
                else:
                    result_label.config(text='❌ 连接失败!', fg='red')

                camera.stop()

                # 3秒后关闭
                test_dialog.after(3000, test_dialog.destroy)

            except Exception as e:
                result_label.config(text=f'❌ 错误: {str(e)}', fg='red')
                test_dialog.after(5000, test_dialog.destroy)

        thread = threading.Thread(target=test, daemon=True)
        thread.start()

    def _on_ok(self):
        """确定按钮"""
        self.result = self._get_config()
        self.dialog.destroy()

    def _on_cancel(self):
        """取消按钮"""
        self.result = None
        self.dialog.destroy()

    def show(self) -> Optional[Dict]:
        """显示对话框"""
        self.dialog.wait_window()
        return self.result

class WoodDefectGUI:
    """木材缺陷检测系统主窗口"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('木材缺陷检测系统')
        self.root.geometry('1400x900')

        # 数据
        self.current_mode = 'batch'
        self.uploaded_image_path: Optional[str] = None
        self.batch_image_paths: List[str] = []
        self.inference_results: Dict[str, InferenceResult] = {}
        self.batch_results: List[BatchResult] = []
        self.dataset_type = 'rubber'

        # 本地摄像头
        self.local_camera: Optional[CameraCapture] = None
        self.local_camera_running = False
        self.local_camera_frame: Optional[np.ndarray] = None

        # 本地摄像头实时检测
        self.local_realtime_results: List[BatchResult] = []
        self.local_realtime_mode = False  # 实时检测模式
        self.local_realtime_interval = 2000  # 实时检测间隔(毫秒)
        self.local_realtime_timer = None  # 定时器
        self.local_detection_count = 0  # 检测计数

        # IP 摄像头
        self.ip_camera: Optional[IPCameraCapture] = None
        self.ip_camera_running = False
        self.ip_camera_frame: Optional[np.ndarray] = None
        self.ip_camera_config: Optional[Dict] = None

        # 连续采集
        self.continuous_mode = False
        self.continuous_interval = 2000  # 2秒
        self.continuous_timer = None
        self.capture_count = 0

        # 视频处理相关变量 (添加在IP摄像头变量之后)
        self.video_capture: Optional[VideoCapture] = None
        self.video_running = False
        self.video_frame: Optional[np.ndarray] = None
        self.video_path: Optional[str] = None

        # 视频检测控制
        self.video_detection_mode = False  # 自动检测模式
        self.video_detection_interval = 30  # 检测间隔(帧数)
        self.video_detection_results: List[BatchResult] = []
        self.video_frame_counter = 0
        self.video_detection_count = 0

        # 自动保存
        self.auto_save = False
        self.save_folder = None

        # 模型配置
        self.available_models = {
            'lam': {'name': 'LAM (Ours)', 'color': '#3b82f6'},
            'unet': {'name': 'U-Net', 'color': '#10b981'},
            'fcn': {'name': 'FCN', 'color': '#f59e0b'},
            'deeplabv3': {'name': 'DeepLabV3', 'color': '#8b5cf6'},
            'deeplabv3plus': {'name': 'DeepLabV3+', 'color': '#ec4899'},
        }

        self.model_vars: Dict[str, tk.BooleanVar] = {}

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        style = ttk.Style()
        style.theme_use('clam')

        self.create_header()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)

        self.create_upload_tab()
        self.create_batch_tab()
        self.create_local_camera_tab()
        self.create_video_tab()
        # self.create_ip_camera_tab()  # 新的 IP 摄像头标签页
        self.create_inference_tab()
        self.create_results_tab()

        self.create_statusbar()

    def create_header(self):
        """创建头部"""
        header_frame = tk.Frame(self.root, bg='#2563eb', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text='木材缺陷检测系统',
            font=('微软雅黑', 20, 'bold'),
            bg='#2563eb',
            fg='white'
        )
        title_label.pack(side='left', padx=20, pady=10)

        control_frame = tk.Frame(header_frame, bg='#2563eb')
        control_frame.pack(side='right', padx=20)

        tk.Label(
            control_frame,
            text='数据集:',
            bg='#2563eb',
            fg='white',
            font=('微软雅黑', 11)
        ).pack(side='left', padx=5)

        self.dataset_var = tk.StringVar(value='rubber')
        dataset_combo = ttk.Combobox(
            control_frame,
            textvariable=self.dataset_var,
            values=['rubber', 'pine'],
            state='readonly',
            width=15
        )
        dataset_combo.pack(side='left', padx=5)
        dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_changed)

        clear_btn = tk.Button(
            control_frame,
            text='清除',
            command=self.clear_all,
            bg='#ef4444',
            fg='white',
            font=('微软雅黑', 10, 'bold'),
            padx=15,
            pady=5
        )
        clear_btn.pack(side='left', padx=5)

    def create_upload_tab(self):
        """创建单图上传标签页"""
        upload_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(upload_frame, text='单图上传')

        center_frame = tk.Frame(upload_frame, bg='white')
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        preview_frame = tk.Frame(center_frame, bg='#f3f4f6', relief='solid', borderwidth=2)
        preview_frame.pack(pady=20)

        self.image_preview_label = tk.Label(
            preview_frame,
            text='点击下方按钮上传图片\n支持 JPG, PNG 格式\n推荐尺寸 512×512',
            font=('微软雅黑', 12),
            bg='#f3f4f6',
            fg='#6b7280',
            width=80,
            height=30
        )
        self.image_preview_label.pack(padx=30, pady=30)

        upload_btn = tk.Button(
            center_frame,
            text='选择图片',
            command=self.upload_single_image,
            bg='#3b82f6',
            fg='white',
            font=('微软雅黑', 14, 'bold'),
            padx=30,
            pady=10,
            cursor='hand2'
        )
        upload_btn.pack(pady=10)

        self.filename_label = tk.Label(
            center_frame,
            text='',
            font=('微软雅黑', 10),
            bg='white',
            fg='#6b7280'
        )
        self.filename_label.pack(pady=5)

        next_btn = tk.Button(
            center_frame,
            text='结果推断',
            command=self.start_single_inference,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 12, 'bold'),
            padx=20,
            pady=8
        )
        next_btn.pack(pady=20)

    def create_batch_tab(self):
        """创建批量处理标签页"""
        batch_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(batch_frame, text='批量处理')

        center_frame = tk.Frame(batch_frame, bg='white')
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        title = tk.Label(
            center_frame,
            text='批量图片处理',
            font=('微软雅黑', 18, 'bold'),
            bg='white'
        )
        title.pack(pady=20)

        btn_frame = tk.Frame(center_frame, bg='white')
        btn_frame.pack(pady=20)

        folder_btn = tk.Button(
            btn_frame,
            text='选择文件夹',
            command=self.select_folder,
            bg='#3b82f6',
            fg='white',
            font=('微软雅黑', 14, 'bold'),
            padx=30,
            pady=15
        )
        folder_btn.pack(side='left', padx=10)

        files_btn = tk.Button(
            btn_frame,
            text='选择多个文件',
            command=self.select_multiple_files,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 14, 'bold'),
            padx=30,
            pady=15
        )
        files_btn.pack(side='left', padx=10)

        self.batch_info_label = tk.Label(
            center_frame,
            text='',
            font=('微软雅黑', 11),
            bg='white',
            fg='#3b82f6'
        )
        self.batch_info_label.pack(pady=10)

        next_btn = tk.Button(
            center_frame,
            text='结果推断',
            command=self.start_batch_processing,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 12, 'bold'),
            padx=20,
            pady=8
        )
        next_btn.pack(pady=10)

    def create_local_camera_tab(self):
        """创建本地摄像头标签页"""
        camera_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(camera_frame, text='本地摄像头')

        # 左侧预览区
        left_frame = tk.Frame(camera_frame, bg='white')
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        preview_label = tk.Label(
            left_frame,
            text='本地摄像头预览',
            font=('微软雅黑', 14, 'bold'),
            bg='white'
        )
        preview_label.pack(pady=10)

        # self.local_camera_preview_label = tk.Label(
        #     left_frame,
        #     text='本地摄像头未启动',
        #     bg='#f3f4f6',
        #     width=80,
        #     height=30
        # )
        # self.local_camera_preview_label.pack(pady=10)

        # 使用Frame来精确控制像素尺寸
        preview_container = tk.Frame(
            left_frame,
            bg='#f3f4f6',
            width=720,  # 像素宽度
            height=650,  # 像素高度
            relief='solid',
            borderwidth=2
        )
        preview_container.pack(pady=10)
        preview_container.pack_propagate(False)  # 关键!防止自动缩放

        self.local_camera_preview_label = tk.Label(
            preview_container,
            text='本地摄像头未启动',
            bg='#f3f4f6',
            fg='#6b7280',
            font=('微软雅黑', 12)
        )
        self.local_camera_preview_label.pack(fill='both', expand=True)

        # 【新增】检测计数和FPS显示
        info_frame = tk.Frame(left_frame, bg='white'
        )
        info_frame.pack(pady=5)
        self.local_detection_count_label = tk.Label(
            info_frame,
            text='已检测: 0',
            font=('微软雅黑', 10),
            bg='white',
            fg='#3b82f6'
        )
        self.local_detection_count_label.pack(side='left', padx=10)

        # 右侧控制区
        right_frame = tk.Frame(camera_frame, bg='white', width=300)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)
        right_frame.pack_propagate(False)

        control_label = tk.Label(
            right_frame,
            text='控制面板',
            font=('微软雅黑', 14, 'bold'),
            bg='white'
        )
        control_label.pack(pady=10)

        # 摄像头选择
        cam_select_frame = tk.Frame(right_frame, bg='white')
        cam_select_frame.pack(pady=5, fill='x')

        tk.Label(
            cam_select_frame,
            text='选择摄像头:',
            font=('微软雅黑', 10),
            bg='white'
        ).pack(side='left')

        self.camera_id_var = tk.StringVar(value='0')
        camera_combo = ttk.Combobox(
            cam_select_frame,
            textvariable=self.camera_id_var,
            values=['0', '1', '2'],
            state='readonly',
            width=10
        )
        camera_combo.pack(side='left', padx=5)

        self.local_camera_start_btn = tk.Button(
            right_frame,
            text=' 启动摄像头',
            command=self.start_local_camera,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 12, 'bold'),
            padx=20,
            pady=10
        )
        self.local_camera_start_btn.pack(pady=10, fill='x')

        self.local_camera_stop_btn = tk.Button(
            right_frame,
            text=' 停止摄像头',
            command=self.stop_local_camera,
            bg='#ef4444',
            fg='white',
            font=('微软雅黑', 12, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.local_camera_stop_btn.pack(pady=10, fill='x')

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # 【新增】检测控制区域
        tk.Label(
            right_frame,
            text='检测控制',
            font=('微软雅黑', 12, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        self.local_camera_capture_btn = tk.Button(
            right_frame,
            text='单次检测',
            command=self.capture_and_infer_local,
            bg='#3b82f6',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.local_camera_capture_btn.pack(pady=5, fill='x')

        # 【新增】实时检测按钮
        self.local_realtime_btn = tk.Button(
            right_frame,
            text='开启实时检测',
            command=self.toggle_local_realtime_mode,
            bg='#8b5cf6',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.local_realtime_btn.pack(pady=5, fill='x')

        # 【新增】实时检测间隔设置
        interval_frame = tk.Frame(right_frame, bg='white')
        interval_frame.pack(pady=5, fill='x')

        tk.Label(
            interval_frame,
            text='间隔(秒):',
            font=('微软雅黑', 9),
            bg='white'
        ).pack(side='left')

        self.local_interval_var = tk.StringVar(value='2')
        interval_spin = tk.Spinbox(
            interval_frame,
            from_=1,
            to=60,
            textvariable=self.local_interval_var,
            width=10,
            font=('微软雅黑', 9)
        )
        interval_spin.pack(side='left', padx=5)

        # 【新增】自动保存选项
        self.local_auto_save_var = tk.BooleanVar(value=False)
        auto_save_cb = tk.Checkbutton(
            right_frame,
            text='自动保存结果',
            variable=self.local_auto_save_var,
            font=('微软雅黑', 10),
            bg='white'
        )
        auto_save_cb.pack(pady=5)

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15)

        # 使用说明
        tk.Label(
            right_frame,
            text=' 使用说明',
            font=('微软雅黑', 11, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        usage_text = """
    1. 选择摄像头编号
       (0=默认, 1=外接)

    2. 点击"启动摄像头"

    3. 单次检测:手动点击检测

    4. 实时检测:自动定时检测
       可设置检测间隔

    5. 开启"自动保存"可保存
       所有检测结果
        """

        # tk.Label(
        #     right_frame,
        #     text=usage_text,
        #     font=('微软雅黑', 8),
        #     bg='white',
        #     fg='#6b7280',
        #     justify='left'
        # ).pack(pady=5)

    def create_ip_camera_tab(self):
        """创建 IP 摄像头标签页"""
        camera_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(camera_frame, text='IP摄像头')

        # 左侧预览区
        left_frame = tk.Frame(camera_frame, bg='white')
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        preview_label = tk.Label(
            left_frame,
            text='IP 摄像头预览',
            font=('微软雅黑', 14, 'bold'),
            bg='white'
        )
        preview_label.pack(pady=10)

        self.ip_camera_preview_label = tk.Label(
            left_frame,
            text='IP 摄像头未启动',
            bg='#f3f4f6',
            width=80,
            height=30
        )
        self.ip_camera_preview_label.pack(pady=10)

        # FPS 和状态显示
        info_frame = tk.Frame(left_frame, bg='white')
        info_frame.pack(pady=5)

        self.fps_label = tk.Label(
            info_frame,
            text='FPS: 0.0',
            font=('微软雅黑', 10),
            bg='white',
            fg='#10b981'
        )
        self.fps_label.pack(side='left', padx=10)

        self.capture_count_label = tk.Label(
            info_frame,
            text='已采集: 0',
            font=('微软雅黑', 10),
            bg='white',
            fg='#3b82f6'
        )
        self.capture_count_label.pack(side='left', padx=10)

        # 右侧控制区
        right_frame = tk.Frame(camera_frame, bg='white', width=350)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)
        right_frame.pack_propagate(False)

        control_label = tk.Label(
            right_frame,
            text='控制面板',
            font=('微软雅黑', 14, 'bold'),
            bg='white'
        )
        control_label.pack(pady=10)

        # 配置按钮
        config_btn = tk.Button(
            right_frame,
            text='配置摄像头',
            command=self.config_ip_camera,
            bg='#f59e0b',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10
        )
        config_btn.pack(pady=5, fill='x')

        # 启动/停止按钮
        self.ip_camera_start_btn = tk.Button(
            right_frame,
            text='启动摄像头',
            command=self.start_ip_camera,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10
        )
        self.ip_camera_start_btn.pack(pady=5, fill='x')

        self.ip_camera_stop_btn = tk.Button(
            right_frame,
            text='停止摄像头',
            command=self.stop_ip_camera,
            bg='#ef4444',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.ip_camera_stop_btn.pack(pady=5, fill='x')

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # 采集控制
        tk.Label(
            right_frame,
            text='采集控制',
            font=('微软雅黑', 12, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        self.ip_camera_capture_btn = tk.Button(
            right_frame,
            text='📸 单次捕获',
            command=self.capture_and_infer_ip,
            bg='#3b82f6',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.ip_camera_capture_btn.pack(pady=5, fill='x')

        self.continuous_btn = tk.Button(
            right_frame,
            text='🔄 开启连续采集',
            command=self.toggle_continuous_mode,
            bg='#8b5cf6',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.continuous_btn.pack(pady=5, fill='x')

        # 连续采集设置
        interval_frame = tk.Frame(right_frame, bg='white')
        interval_frame.pack(pady=5, fill='x')

        tk.Label(
            interval_frame,
            text='间隔(秒):',
            font=('微软雅黑', 9),
            bg='white'
        ).pack(side='left')

        self.interval_var = tk.StringVar(value='2')
        interval_spin = tk.Spinbox(
            interval_frame,
            from_=1,
            to=60,
            textvariable=self.interval_var,
            width=10,
            font=('微软雅黑', 9)
        )
        interval_spin.pack(side='left', padx=5)

        # 自动保存
        self.auto_save_var = tk.BooleanVar(value=False)
        auto_save_cb = tk.Checkbutton(
            right_frame,
            text='自动保存结果',
            variable=self.auto_save_var,
            command=self.toggle_auto_save,
            font=('微软雅黑', 10),
            bg='white'
        )
        auto_save_cb.pack(pady=5)

        # 保存文件夹按钮
        self.save_folder_btn = tk.Button(
            right_frame,
            text='📁 选择保存位置',
            command=self.select_save_folder,
            bg='#6b7280',
            fg='white',
            font=('微软雅黑', 9),
            padx=10,
            pady=5,
            state='disabled'
        )
        self.save_folder_btn.pack(pady=5, fill='x')

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # 使用说明
        tk.Label(
            right_frame,
            text='使用说明',
            font=('微软雅黑', 11, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        usage_text = """
1. 点击"配置摄像头"设置IP摄像头参数
2. 点击"启动摄像头"开始预览
3. 单次捕获：手动点击检测
4. 连续采集：自动定时检测
5. 开启"自动保存"可保存所有结果
        """

        tk.Label(
            right_frame,
            text=usage_text,
            font=('微软雅黑', 8),
            bg='white',
            fg='#6b7280',
            justify='left'
        ).pack(pady=5)

    def create_inference_tab(self):
        """创建推断配置标签页"""
        inference_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(inference_frame, text='模型配置')

        canvas = tk.Canvas(inference_frame, bg='white')
        scrollbar = ttk.Scrollbar(inference_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        model_frame = tk.LabelFrame(
            scrollable_frame,
            text='选择推断模型',
            font=('微软雅黑', 14, 'bold'),
            bg='white',
            padx=20,
            pady=20
        )
        model_frame.pack(fill='x', padx=20, pady=20)

        for idx, (model_id, model_info) in enumerate(self.available_models.items()):
            # var = tk.BooleanVar(value=True)
            var = tk.BooleanVar(value=(model_id == 'lam'))
            self.model_vars[model_id] = var

            cb = tk.Checkbutton(
                model_frame,
                text=model_info['name'],
                variable=var,
                font=('微软雅黑', 12),
                bg='white',
                activebackground='white'
            )
            cb.grid(row=idx // 2, column=idx % 2, sticky='w', padx=10, pady=5)

        preview_frame = tk.LabelFrame(
            scrollable_frame,
            text='输入图片预览',
            font=('微软雅黑', 14, 'bold'),
            bg='white',
            padx=20,
            pady=20
        )
        preview_frame.pack(fill='x', padx=20, pady=20)

        self.inference_preview_label = tk.Label(
            preview_frame,
            text='暂无图片',
            bg='#f3f4f6',
            width=60,
            height=25
        )
        self.inference_preview_label.pack()

        btn_container = tk.Frame(scrollable_frame, bg='white')
        btn_container.pack(pady=30)

        single_inference_btn = tk.Button(
            btn_container,
            text='开始单图推断',
            command=self.start_single_inference,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 16, 'bold'),
            padx=40,
            pady=15,
            cursor='hand2'
        )
        single_inference_btn.pack(side='left', padx=10)

        batch_inference_btn = tk.Button(
            btn_container,
            text='开始批量处理',
            command=self.start_batch_processing,
            bg='#f59e0b',
            fg='white',
            font=('微软雅黑', 16, 'bold'),
            padx=40,
            pady=15,
            cursor='hand2'
        )
        batch_inference_btn.pack(side='left', padx=10)

        self.progress = ttk.Progressbar(
            scrollable_frame,
            length=400,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        self.progress.pack_forget()

    def create_results_tab(self):
        """创建结果标签页"""
        results_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(results_frame, text='结果分析')

        canvas = tk.Canvas(results_frame, bg='white')
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=canvas.yview)
        self.results_scrollable = tk.Frame(canvas, bg='white')

        self.results_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.results_scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        table_frame = tk.LabelFrame(
            self.results_scrollable,
            text='性能对比',
            font=('微软雅黑', 14, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        table_frame.pack(fill='x', padx=20, pady=20)

        columns = ('模型', 'mIoU', 'mAcc', 'F1 Score', '推断时间(ms)')
        self.results_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            height=8
        )

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150, anchor='center')

        self.results_tree.pack(fill='x')

        self.vis_frame = tk.LabelFrame(
            self.results_scrollable,
            text='分割结果可视化',
            font=('微软雅黑', 14, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        self.vis_frame.pack(fill='both', expand=True, padx=20, pady=20)

        export_btn = tk.Button(
            self.results_scrollable,
            text='导出完整报告',
            command=self.export_report,
            bg='#3b82f6',
            fg='white',
            font=('微软雅黑', 12, 'bold'),
            padx=20,
            pady=10
        )
        export_btn.pack(pady=20)

        self.results_manager = ResultsManager(
            self.vis_frame,
            self.results_tree,
            self.available_models,
            self.dataset_type
        )

    def create_statusbar(self):
        """创建状态栏"""
        self.statusbar = tk.Label(
            self.root,
            text='就绪',
            relief='sunken',
            anchor='w',
            bg='#f3f4f6',
            font=('微软雅黑', 9)
        )
        self.statusbar.pack(side='bottom', fill='x')

    def create_video_tab(self):
        """创建视频处理标签页"""
        video_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(video_frame, text='视频检测')

        # 左侧预览区
        left_frame = tk.Frame(video_frame, bg='white')
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        preview_label = tk.Label(
            left_frame,
            text='视频预览',
            font=('微软雅黑', 14, 'bold'),
            bg='white'
        )
        preview_label.pack(pady=10)

        # 视频预览容器
        preview_container = tk.Frame(
            left_frame,
            bg='#f3f4f6',
            width=720,
            height=500,
            relief='solid',
            borderwidth=2
        )
        preview_container.pack(pady=10)
        preview_container.pack_propagate(False)

        self.video_preview_label = tk.Label(
            preview_container,
            text='未加载视频\n点击右侧"选择视频"按钮',
            bg='#f3f4f6',
            fg='#6b7280',
            font=('微软雅黑', 12)
        )
        self.video_preview_label.pack(fill='both', expand=True)

        # 进度条
        self.video_progress = ttk.Progressbar(
            left_frame,
            length=700,
            mode='determinate'
        )
        self.video_progress.pack(pady=5)

        # 视频信息显示
        info_frame = tk.Frame(left_frame, bg='white')
        info_frame.pack(pady=5)

        self.video_info_label = tk.Label(
            info_frame,
            text='',
            font=('微软雅黑', 10),
            bg='white',
            fg='#6b7280'
        )
        self.video_info_label.pack()

        self.video_detection_count_label = tk.Label(
            info_frame,
            text='已检测: 0 帧',
            font=('微软雅黑', 10),
            bg='white',
            fg='#3b82f6'
        )
        self.video_detection_count_label.pack(pady=5)

        # 右侧控制区
        right_frame = tk.Frame(video_frame, bg='white', width=350)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)
        right_frame.pack_propagate(False)

        control_label = tk.Label(
            right_frame,
            text='控制面板',
            font=('微软雅黑', 14, 'bold'),
            bg='white'
        )
        control_label.pack(pady=10)

        # 视频选择
        tk.Label(
            right_frame,
            text='📁 视频文件',
            font=('微软雅黑', 12, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        select_video_btn = tk.Button(
            right_frame,
            text='选择视频文件',
            command=self.select_video_file,
            bg='#3b82f6',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10
        )
        select_video_btn.pack(pady=5, fill='x')

        self.video_filename_label = tk.Label(
            right_frame,
            text='',
            font=('微软雅黑', 9),
            bg='white',
            fg='#6b7280',
            wraplength=300
        )
        self.video_filename_label.pack(pady=5)

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # 播放控制
        tk.Label(
            right_frame,
            text='▶️ 播放控制',
            font=('微软雅黑', 12, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        play_control_frame = tk.Frame(right_frame, bg='white')
        play_control_frame.pack(pady=5, fill='x')

        self.video_play_btn = tk.Button(
            play_control_frame,
            text='▶️ 播放',
            command=self.play_video,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 10, 'bold'),
            padx=10,
            pady=8,
            state='disabled'
        )
        self.video_play_btn.pack(side='left', padx=2, expand=True, fill='x')

        self.video_pause_btn = tk.Button(
            play_control_frame,
            text='⏸️ 暂停',
            command=self.pause_video,
            bg='#f59e0b',
            fg='white',
            font=('微软雅黑', 10, 'bold'),
            padx=10,
            pady=8,
            state='disabled'
        )
        self.video_pause_btn.pack(side='left', padx=2, expand=True, fill='x')

        self.video_stop_btn = tk.Button(
            right_frame,
            text='️ 停止',
            command=self.stop_video,
            bg='#ef4444',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.video_stop_btn.pack(pady=5, fill='x')

        # 播放速度
        speed_frame = tk.Frame(right_frame, bg='white')
        speed_frame.pack(pady=5, fill='x')

        tk.Label(
            speed_frame,
            text='速度:',
            font=('微软雅黑', 9),
            bg='white'
        ).pack(side='left')

        self.video_speed_var = tk.StringVar(value='1.0')
        speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self.video_speed_var,
            values=['0.5', '0.75', '1.0', '1.5', '2.0'],
            state='readonly',
            width=10
        )
        speed_combo.pack(side='left', padx=5)
        speed_combo.bind('<<ComboboxSelected>>', self.on_speed_changed)

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # 检测控制
        tk.Label(
            right_frame,
            text='🔍 检测控制',
            font=('微软雅黑', 12, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        self.video_capture_btn = tk.Button(
            right_frame,
            text='📸 捕获当前帧',
            command=self.capture_video_frame,
            bg='#8b5cf6',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.video_capture_btn.pack(pady=5, fill='x')

        self.video_auto_detect_btn = tk.Button(
            right_frame,
            text='🤖 开启自动检测',
            command=self.toggle_video_auto_detect,
            bg='#ec4899',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.video_auto_detect_btn.pack(pady=5, fill='x')

        # 检测间隔设置
        interval_frame = tk.Frame(right_frame, bg='white')
        interval_frame.pack(pady=5, fill='x')

        tk.Label(
            interval_frame,
            text='间隔(帧):',
            font=('微软雅黑', 9),
            bg='white'
        ).pack(side='left')

        self.video_interval_var = tk.StringVar(value='30')
        interval_spin = tk.Spinbox(
            interval_frame,
            from_=1,
            to=300,
            textvariable=self.video_interval_var,
            width=10,
            font=('微软雅黑', 9)
        )
        interval_spin.pack(side='left', padx=5)

        # 批量提取
        extract_btn = tk.Button(
            right_frame,
            text='📊 批量提取帧',
            command=self.extract_video_frames,
            bg='#06b6d4',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        extract_btn.pack(pady=5, fill='x')
        self.video_extract_btn = extract_btn

        # 自动保存
        self.video_auto_save_var = tk.BooleanVar(value=False)
        auto_save_cb = tk.Checkbutton(
            right_frame,
            text='自动保存检测结果',
            variable=self.video_auto_save_var,
            font=('微软雅黑', 10),
            bg='white'
        )
        auto_save_cb.pack(pady=5)

        # 分隔线
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # 使用说明
        tk.Label(
            right_frame,
            text='💡 使用说明',
            font=('微软雅黑', 11, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        usage_text = """
    1. 选择视频文件
    2. 播放预览视频
    3. 单帧检测: 暂停后点击"捕获当前帧"
    4. 自动检测: 播放时开启自动检测
    5. 批量提取: 按间隔提取帧并检测
        """

        tk.Label(
            right_frame,
            text=usage_text,
            font=('微软雅黑', 8),
            bg='white',
            fg='#6b7280',
            justify='left'
        ).pack(pady=5)

    # ==================== 本地摄像头 ====================

    def start_local_camera(self):
        """启动本地摄像头"""
        camera_id = int(self.camera_id_var.get())

        if not self.local_camera:
            self.local_camera = CameraCapture(camera_id=camera_id)
            self.local_camera.set_callbacks(
                self.on_local_camera_frame,
                self.on_local_camera_error
            )

        if self.local_camera.start():
            self.local_camera_running = True
            self.local_camera_start_btn.config(state='disabled')
            self.local_camera_stop_btn.config(state='normal')
            self.local_camera_capture_btn.config(state='normal')
            self.local_realtime_btn.config(state='normal')  # 【新增】启用实时检测按钮
            self.update_status(' 本地摄像头已启动')

            self.update_local_camera_preview()
        else:
            messagebox.showerror('错误', '无法启动本地摄像头')

    def stop_local_camera(self):
        """停止本地摄像头"""
        # 【新增】停止实时检测
        if self.local_realtime_mode:
            self.toggle_local_realtime_mode()

        if self.local_camera:
            self.local_camera.stop()
            self.local_camera_running = False
            self.local_camera_start_btn.config(state='normal')
            self.local_camera_stop_btn.config(state='disabled')
            self.local_camera_capture_btn.config(state='disabled')
            self.local_realtime_btn.config(state='disabled')  # 【新增】禁用实时检测按钮
            self.local_camera_preview_label.config(text='本地摄像头已停止', image='')
            self.update_status(' 本地摄像头已停止')

    def on_local_camera_frame(self, frame: np.ndarray):
        """本地摄像头帧回调"""
        self.local_camera_frame = frame

    def on_local_camera_error(self, error_msg: str):
        """本地摄像头错误"""
        messagebox.showerror('摄像头错误', error_msg)
        self.stop_local_camera()

    def update_local_camera_preview(self):
        """更新本地摄像头预览"""
        if self.local_camera_running and self.local_camera_frame is not None:
            image = Image.fromarray(self.local_camera_frame)
            image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            self.local_camera_preview_label.config(image=photo, text='')
            self.local_camera_preview_label.image = photo

        if self.local_camera_running:
            self.root.after(30, self.update_local_camera_preview)

    def capture_and_infer_local(self):
        """捕获并推断（本地摄像头）"""
        if not self.local_camera or not self.local_camera_running:
            messagebox.showwarning('警告', '请先启动本地摄像头!')
            return

        snapshot = self.local_camera.capture_snapshot()
        if snapshot is None:
            messagebox.showwarning('警告', '无法捕获图像!')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = f'temp_local_camera_{timestamp}.jpg'
        Image.fromarray(snapshot).save(temp_path)

        self.current_mode = 'camera'
        self.uploaded_image_path = temp_path

        image = Image.fromarray(snapshot)
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.inference_preview_label.config(image=photo, text='')
        self.inference_preview_label.image = photo

        tab_index = 4 if IPCameraCapture else 3
        self.notebook.select(tab_index)
        self.update_status('已捕获图像，准备推断...')

        self.root.after(500, self.start_single_inference)

    # ==================== 新增5: 实时检测相关方法 ====================
    def toggle_local_realtime_mode(self):
        """切换本地摄像头实时检测模式"""
        if not self.local_realtime_mode:
            # 开启实时检测
            self.local_realtime_mode = True
            self.local_realtime_interval = int(float(self.local_interval_var.get()) * 1000)
            self.local_realtime_btn.config(text=' 停止实时检测', bg='#ef4444')
            self.local_detection_count = 0
            self.local_detection_count_label.config(text=f'已检测: 0')
            self.local_realtime_results = []
            self.update_status(f' 实时检测已启动 (间隔: {self.local_realtime_interval / 1000}秒)')

            # 开始实时检测
            self.do_local_realtime_detection()
        else:
            # 停止实时检测
            self.local_realtime_mode = False
            if self.local_realtime_timer:
                self.root.after_cancel(self.local_realtime_timer)
                self.local_realtime_timer = None
            self.local_realtime_btn.config(text=' 开启实时检测', bg='#8b5cf6')
            if self.local_realtime_results:
                self.results_manager.display_batch_results(self.local_realtime_results)
                self.notebook.select(4 if IPCameraCapture else 3)  # 跳转到结果分析页
                self.update_status(f'✅ 实时检测已停止 (共检测 {self.local_detection_count} 次, 查看结果分析页)')
            else:
                self.update_status(f'️ 实时检测已停止 (共检测 {self.local_detection_count} 次)')

    def do_local_realtime_detection(self):
        """执行本地摄像头实时检测"""
        if not self.local_realtime_mode:
            return

        # 执行检测
        self.local_detection_count += 1
        self.local_detection_count_label.config(text=f'已检测: {self.local_detection_count}')

        # 调用检测方法(静默模式，不弹窗)
        self.capture_and_infer_local_silent()

        # 设置下次检测
        self.local_realtime_timer = self.root.after(
            self.local_realtime_interval,
            self.do_local_realtime_detection
        )

    def capture_and_infer_local_silent(self):
        """捕获并推断(本地摄像头) - 静默模式，用于实时检测"""
        if not self.local_camera or not self.local_camera_running:
            return

        snapshot = self.local_camera.capture_snapshot()
        if snapshot is None:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 如果开启自动保存
        if self.local_auto_save_var.get() and self.save_folder:
            temp_path = os.path.join(self.save_folder, f'local_detect_{timestamp}.jpg')
        else:
            temp_path = f'temp_local_camera_{timestamp}.jpg'

        Image.fromarray(snapshot).save(temp_path)

        self.current_mode = 'camera'
        self.uploaded_image_path = temp_path

        # 更新预览(小图) - 使用 after 确保在主线程
        def update_preview():
            image = Image.fromarray(snapshot)
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.inference_preview_label.config(image=photo, text='')
            self.inference_preview_label.image = photo

        self.root.after(0, update_preview)

        self.update_status(f'✅ 实时检测 #{self.local_detection_count}')

        # 延迟开始推断
        self.root.after(50, self.start_single_inference_silent)

    def start_single_inference_silent(self):
        """开始单图推断 - 静默模式"""
        print("开始单图推断")
        if not self.uploaded_image_path:
            return

        selected_models = [
            model_id for model_id, var in self.model_vars.items()
            if var.get()
        ]

        if not selected_models:
            return

        worker = InferenceWorker(
            self.uploaded_image_path,
            selected_models,
            self.dataset_type
        )

        worker.set_callbacks(
            lambda v, m: None,  # 不更新进度条
            self.on_finished_silent,  # 静默完成回调
            lambda e: None  # 忽略错误
        )

        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()

    def on_finished_silent(self, results: Dict[str, InferenceResult]):
        """推断完成 - 静默模式"""
        print("=== 开始 on_finished_silent ===")

        # 检查 results 是否为空
        if not results:
            print("❌❌❌ results 为空，无法创建 BatchResult")
            return

        # 检查每个模型的结果
        avg_t = 0
        for model_id, result in results.items():
            print(f"模型 {model_id}: {result}")
            if hasattr(result, 'metrics'):
                print(f"  指标: {result.metrics}")
            if hasattr(result, 'inference_time'):
                avg_t += result.inference_time
                print(f"  推断时间: {result.inference_time}")

        try:
            batch_result = BatchResult(
                image_path=self.uploaded_image_path,
                results=results,
                avg_time=avg_t/len(results)
            )
            print("✅ BatchResult 创建成功:")
            print(f"  图片路径: {batch_result.image_path}")
            print(f"  结果数量: {len(batch_result.results)}")

        except Exception as e:
            print(f"❌❌❌ 创建 BatchResult 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        self.local_realtime_results.append(batch_result)
        print(f"✅ 已添加到实时结果列表，当前总数: {len(self.local_realtime_results)}")

        # 使用 after 确保在主线程中更新GUI
        def update_gui():
            print("=== 开始 GUI 更新 ===")
            try:
                if len(self.local_realtime_results) > 0:
                    print(f"准备显示批量结果，共 {len(self.local_realtime_results)} 个结果")
                    self.results_manager.display_batch_results(self.local_realtime_results)
                    print("✅ 批量结果显示完成")
                else:
                    print("❌ 实时结果列表为空")

                # 如果开启自动保存，保存结果
                if self.local_auto_save_var.get() and self.save_folder:
                    print("开始自动保存结果...")
                    self.save_detection_results_silent(results)
                    print("✅ 自动保存完成")

                if results:
                    avg_time = np.mean([r.inference_time for r in results.values()])
                    status_msg = f'✅ 检测#{self.local_detection_count}完成 ({avg_time:.1f}ms)'
                    self.update_status(status_msg)
                    print(f"✅ 状态更新: {status_msg}")
                else:
                    print("❌ 无法计算平均时间，results 为空")

            except Exception as e:
                print(f"❌❌❌ GUI 更新错误: {str(e)}")
                import traceback
                traceback.print_exc()

        # 在主线程中执行GUI更新
        self.root.after(0, update_gui)
        print("=== on_finished_silent 结束 ===")

    def save_detection_results_silent(self, results: Dict[str, InferenceResult]):
        """静默保存检测结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存每个模型的结果图
            for model_id, result in results.items():
                if result.visualization is not None:
                    vis_path = os.path.join(
                        self.save_folder,
                        f'result_{model_id}_{timestamp}.jpg'
                    )
                    Image.fromarray(result.visualization).save(vis_path)

        except Exception as e:
            print(f"保存结果失败: {str(e)}")

    # ==================== IP 摄像头功能 ====================

    def config_ip_camera(self):
        """配置 IP 摄像头"""
        dialog = IPCameraConfigDialog(self.root)
        config = dialog.show()

        if config:
            self.ip_camera_config = config
            self.update_status(f'IP摄像头配置已保存')
            messagebox.showinfo('成功', '摄像头配置已保存\n请点击"启动摄像头"')

    def start_ip_camera(self):
        """启动 IP 摄像头"""
        if not self.ip_camera_config:
            messagebox.showwarning('警告', '请先配置 IP 摄像头!')
            self.config_ip_camera()
            return

        if self.ip_camera_running:
            messagebox.showinfo('提示', '摄像头已在运行中')
            return

        try:
            # 创建摄像头实例
            config = self.ip_camera_config
            if config['mode'] == 'template':
                self.ip_camera = IPCameraCapture(
                    camera_type=config['camera_type'],
                    ip=config['ip'],
                    port=config['port'],
                    user=config['user'],
                    password=config['password'],
                    channel=config['channel']
                )
            else:
                self.ip_camera = IPCameraCapture(
                    camera_url=config['camera_url']
                )

            # 设置回调
            self.ip_camera.set_callbacks(
                self.on_ip_camera_frame,
                self.on_ip_camera_error
            )

            # 启动
            if self.ip_camera.start():
                self.ip_camera_running = True
                self.ip_camera_start_btn.config(state='disabled')
                self.ip_camera_stop_btn.config(state='normal')
                self.ip_camera_capture_btn.config(state='normal')
                self.continuous_btn.config(state='normal')
                self.update_status(' IP 摄像头已启动')

                # 开始更新预览
                self.update_ip_camera_preview()
                # 开始更新 FPS
                self.update_fps_display()
            else:
                messagebox.showerror('错误', '无法启动 IP 摄像头')

        except Exception as e:
            messagebox.showerror('错误', f'启动失败: {str(e)}')
            self.update_status('IP 摄像头启动失败')

    def stop_ip_camera(self):
        """停止 IP 摄像头"""
        if self.ip_camera:
            # 停止连续采集
            if self.continuous_mode:
                self.toggle_continuous_mode()

            self.ip_camera.stop()
            self.ip_camera_running = False
            self.ip_camera_start_btn.config(state='normal')
            self.ip_camera_stop_btn.config(state='disabled')
            self.ip_camera_capture_btn.config(state='disabled')
            self.continuous_btn.config(state='disabled')
            self.ip_camera_preview_label.config(text='IP 摄像头已停止', image='')
            self.update_status('IP 摄像头已停止')

    def on_ip_camera_frame(self, frame: np.ndarray):
        """IP 摄像头帧回调"""
        self.ip_camera_frame = frame

    def on_ip_camera_error(self, error_msg: str):
        """IP 摄像头错误回调"""
        messagebox.showerror('摄像头错误', error_msg)
        self.stop_ip_camera()

    def update_ip_camera_preview(self):
        """更新 IP 摄像头预览"""
        if self.ip_camera_running and self.ip_camera:
            frame = self.ip_camera.get_latest_frame()
            if frame is not None:
                image = Image.fromarray(frame)
                image.thumbnail((800, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)

                self.ip_camera_preview_label.config(image=photo, text='')
                self.ip_camera_preview_label.image = photo

        if self.ip_camera_running:
            self.root.after(30, self.update_ip_camera_preview)

    def update_fps_display(self):
        """更新 FPS 显示"""
        if self.ip_camera_running and self.ip_camera:
            fps = self.ip_camera.get_fps()
            self.fps_label.config(text=f'FPS: {fps:.1f}')

        if self.ip_camera_running:
            self.root.after(1000, self.update_fps_display)

    def capture_and_infer_ip(self):
        """捕获并推断（IP 摄像头）"""
        if not self.ip_camera or not self.ip_camera_running:
            messagebox.showwarning('警告', '请先启动 IP 摄像头!')
            return

        snapshot = self.ip_camera.capture_snapshot()
        if snapshot is None:
            messagebox.showwarning('警告', '无法捕获图像!')
            return

        # 保存为临时文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = f'temp_ip_camera_{timestamp}.jpg'
        Image.fromarray(snapshot).save(temp_path)

        self.current_mode = 'camera'
        self.uploaded_image_path = temp_path

        # 更新预览
        image = Image.fromarray(snapshot)
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.inference_preview_label.config(image=photo, text='')
        self.inference_preview_label.image = photo

        # 跳转到推断页
        self.notebook.select(3)
        self.update_status(f'✅ 已捕获图像 #{self.capture_count + 1}')

        # 延迟开始推断
        self.root.after(300, self.start_single_inference)

    def toggle_continuous_mode(self):
        """切换连续采集模式"""
        if not self.continuous_mode:
            # 开启连续采集
            self.continuous_mode = True
            self.continuous_interval = int(float(self.interval_var.get()) * 1000)
            self.continuous_btn.config(text=' 停止连续采集', bg='#ef4444')
            self.capture_count = 0
            self.update_status(f' 连续采集已启动 (间隔: {self.continuous_interval / 1000}秒)')

            # 开始连续采集
            self.do_continuous_capture()
        else:
            # 停止连续采集
            self.continuous_mode = False
            if self.continuous_timer:
                self.root.after_cancel(self.continuous_timer)
                self.continuous_timer = None
            self.continuous_btn.config(text=' 开启连续采集', bg='#8b5cf6')
            self.update_status(f'️ 连续采集已停止 (共采集 {self.capture_count} 次)')

    def do_continuous_capture(self):
        """执行连续采集"""
        if not self.continuous_mode:
            return

        # 捕获并推断
        self.capture_count += 1
        self.capture_count_label.config(text=f'已采集: {self.capture_count}')
        self.capture_and_infer_ip()

        # 设置下次采集
        self.continuous_timer = self.root.after(
            self.continuous_interval,
            self.do_continuous_capture
        )

    def toggle_auto_save(self):
        """切换自动保存"""
        if self.auto_save_var.get():
            self.save_folder_btn.config(state='normal')
            if not self.save_folder:
                self.select_save_folder()
        else:
            self.save_folder_btn.config(state='disabled')

    def select_save_folder(self):
        """选择保存文件夹"""
        folder = filedialog.askdirectory(title='选择结果保存文件夹')
        if folder:
            self.save_folder = folder
            self.update_status(f'📁 保存位置: {folder}')
            messagebox.showinfo('成功', f'结果将保存到:\n{folder}')

    # ==================== 视频处理方法 ====================
    def select_video_file(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            from GUI.Work.VideoCapture import VideoCapture

            self.video_path = file_path
            self.video_capture = VideoCapture(file_path)

            # 设置回调
            self.video_capture.set_callbacks(
                frame_cb=self.on_video_frame,
                error_cb=self.on_video_error,
                progress_cb=self.on_video_progress
            )

            # 加载视频
            if self.video_capture.load_video(file_path):
                info = self.video_capture.get_info()
                filename = os.path.basename(file_path)

                self.video_filename_label.config(text=f'已选择: {filename}')
                self.video_info_label.config(
                    text=f"{info['width']}x{info['height']} | {info['fps']:.1f}fps | "
                         f"{info['total_frames']}帧 | {info['duration']:.1f}秒"
                )

                # 启用按钮
                self.video_play_btn.config(state='normal')
                self.video_extract_btn.config(state='normal')

                self.update_status(f'✅ 视频已加载: {filename}')

                # 显示第一帧
                first_frame = self.video_capture.capture_current_frame()
                if first_frame is not None:
                    self.display_video_frame(first_frame)

    def play_video(self):
        """播放视频"""
        if self.video_capture:
            if self.video_capture.start():
                self.video_running = True
                self.video_play_btn.config(state='disabled')
                self.video_pause_btn.config(state='normal')
                self.video_stop_btn.config(state='normal')
                self.video_capture_btn.config(state='normal')
                self.video_auto_detect_btn.config(state='normal')
                self.update_video_preview()
                self.update_status('️ 视频播放中...')

    def pause_video(self):
        """暂停视频"""
        if self.video_capture:
            self.video_capture.pause()
            self.video_pause_btn.config(text='️ 继续', bg='#10b981')
            self.video_pause_btn.config(command=self.resume_video)
            self.update_status('️ 视频已暂停')

    def resume_video(self):
        """恢复播放"""
        if self.video_capture:
            self.video_capture.resume()
            self.video_pause_btn.config(text='️ 暂停', bg='#f59e0b')
            self.video_pause_btn.config(command=self.pause_video)
            self.update_status('️ 继续播放...')

    def stop_video(self):
        """停止视频"""
        if self.video_detection_mode:
            self.toggle_video_auto_detect()

        if self.video_capture:
            self.video_capture.stop()
            self.video_running = False
            self.video_play_btn.config(state='normal')
            self.video_pause_btn.config(state='disabled', text='️ 暂停')
            self.video_pause_btn.config(command=self.pause_video)
            self.video_stop_btn.config(state='disabled')
            self.video_capture_btn.config(state='disabled')
            self.video_auto_detect_btn.config(state='disabled')
            self.video_progress['value'] = 0
            self.update_status('️ 视频已停止')

    def on_speed_changed(self, event):
        """播放速度改变"""
        if self.video_capture:
            speed = float(self.video_speed_var.get())
            self.video_capture.set_playback_speed(speed)

    def on_video_frame(self, frame: np.ndarray):
        """视频帧回调"""
        self.video_frame = frame

        # 自动检测模式
        if self.video_detection_mode:
            self.video_frame_counter += 1
            interval = int(self.video_interval_var.get())

            if self.video_frame_counter >= interval:
                self.video_frame_counter = 0
                self.detect_video_frame_silent(frame)

    def on_video_error(self, error_msg: str):
        """视频错误回调"""
        messagebox.showerror('视频错误', error_msg)

    def on_video_progress(self, current: int, total: int, percentage: float):
        """视频进度回调"""
        self.video_progress['value'] = percentage

    def display_video_frame(self, frame: np.ndarray):
        """显示视频帧"""
        image = Image.fromarray(frame)
        image.thumbnail((700, 480), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        self.video_preview_label.config(image=photo, text='')
        self.video_preview_label.image = photo

    def update_video_preview(self):
        """更新视频预览"""
        if self.video_running and self.video_frame is not None:
            self.display_video_frame(self.video_frame)

        if self.video_running:
            self.root.after(30, self.update_video_preview)

    def capture_video_frame(self):
        """捕获视频当前帧并检测"""
        if not self.video_capture:
            messagebox.showwarning('警告', '请先加载视频!')
            return

        snapshot = self.video_capture.capture_current_frame()
        if snapshot is None:
            messagebox.showwarning('警告', '无法捕获帧!')
            return

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = f'temp_video_frame_{timestamp}.jpg'
        Image.fromarray(snapshot).save(temp_path)

        self.current_mode = 'video'
        self.uploaded_image_path = temp_path

        # 更新预览
        image = Image.fromarray(snapshot)
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.inference_preview_label.config(image=photo, text='')
        self.inference_preview_label.image = photo

        self.notebook.select(4)
        self.update_status(' 已捕获帧，准备检测...')

        self.root.after(500, self.start_single_inference)

    def toggle_video_auto_detect(self):
        """切换自动检测模式"""
        if not self.video_detection_mode:
            self.video_detection_mode = True
            self.video_frame_counter = 0
            self.video_detection_count = 0
            self.video_detection_results = []
            self.video_auto_detect_btn.config(
                text=' 停止自动检测',
                bg='#ef4444'
            )
            self.video_detection_count_label.config(text='已检测: 0 帧')
            self.update_status(' 自动检测已启动')
        else:
            self.video_detection_mode = False
            self.video_auto_detect_btn.config(
                text=' 开启自动检测',
                bg='#ec4899'
            )
            if self.video_detection_results:
                self.results_manager.display_batch_results(self.video_detection_results)
                self.notebook.select(5)
                self.update_status(
                    f' 自动检测已停止 (共检测 {self.video_detection_count} 帧)'
                )

    def detect_video_frame_silent(self, frame: np.ndarray):
        """静默检测视频帧"""
        self.video_detection_count += 1
        self.video_detection_count_label.config(
            text=f'已检测: {self.video_detection_count} 帧'
        )

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        if self.video_auto_save_var.get() and self.save_folder:
            temp_path = os.path.join(
                self.save_folder,
                f'video_frame_{timestamp}.jpg'
            )
        else:
            temp_path = f'temp_video_frame_{timestamp}.jpg'

        Image.fromarray(frame).save(temp_path)

        # 异步检测
        threading.Thread(
            target=self._async_detect_frame,
            args=(temp_path,),
            daemon=True
        ).start()

    def _async_detect_frame(self, image_path: str):
        """异步检测帧"""
        selected_models = [
            model_id for model_id, var in self.model_vars.items()
            if var.get()
        ]

        if not selected_models:
            return

        from GUI.Work.InferenceWorker import InferenceWorker

        worker = InferenceWorker(
            image_path,
            selected_models,
            self.dataset_type
        )
        worker.run()

        if worker.results:
            avg_time = np.mean([r.inference_time for r in worker.results.values()])
            batch_result = BatchResult(
                image_path=image_path,
                results=worker.results,
                avg_time=avg_time
            )
            self.video_detection_results.append(batch_result)

    def extract_video_frames(self):
        """批量提取视频帧"""
        if not self.video_capture:
            messagebox.showwarning('警告', '请先加载视频!')
            return

        # 询问提取参数
        dialog = tk.Toplevel(self.root)
        dialog.title('批量提取设置')
        dialog.geometry('400x250')
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text='批量提取帧设置',
            font=('微软雅黑', 14, 'bold')
        ).pack(pady=20)

        # 间隔设置
        interval_frame = tk.Frame(dialog)
        interval_frame.pack(pady=10)

        tk.Label(interval_frame, text='帧间隔:').pack(side='left')
        interval_var = tk.StringVar(value='30')
        tk.Spinbox(
            interval_frame,
            from_=1,
            to=300,
            textvariable=interval_var,
            width=10
        ).pack(side='left', padx=5)

        # 最大帧数
        max_frame_frame = tk.Frame(dialog)
        max_frame_frame.pack(pady=10)

        tk.Label(max_frame_frame, text='最大提取数:').pack(side='left')
        max_var = tk.StringVar(value='50')
        tk.Spinbox(
            max_frame_frame,
            from_=1,
            to=1000,
            textvariable=max_var,
            width=10
        ).pack(side='left', padx=5)

        result = {'confirmed': False}

        def on_confirm():
            result['confirmed'] = True
            result['interval'] = int(interval_var.get())
            result['max_frames'] = int(max_var.get())
            dialog.destroy()

        tk.Button(
            dialog,
            text='开始提取',
            command=on_confirm,
            bg='#10b981',
            fg='white',
            font=('微软雅黑', 11, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=20)

        dialog.wait_window()

        if result['confirmed']:
            self._do_extract_frames(result['interval'], result['max_frames'])

    def _do_extract_frames(self, interval: int, max_frames: int):
        """执行帧提取"""
        self.update_status(f' 正在提取帧 (间隔: {interval}, 最大: {max_frames})...')

        def extract_thread():
            frames = self.video_capture.extract_frames(interval, max_frames)

            if frames:
                # 保存帧并进行批量检测
                temp_paths = []
                for i, frame in enumerate(frames):
                    temp_path = f'temp_video_extract_{i}.jpg'
                    Image.fromarray(frame).save(temp_path)
                    temp_paths.append(temp_path)

                # 使用批量处理
                self.batch_image_paths = temp_paths
                self.current_mode = 'batch'
                self.root.after(0, self.start_batch_processing)

        threading.Thread(target=extract_thread, daemon=True).start()

    # ==================== 多图推断 ====================

    def upload_single_image(self):
        """上传单张图片"""
        file_path = filedialog.askopenfilename(
            title="选择木材图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.current_mode = 'single'
            self.uploaded_image_path = file_path

            image = Image.open(file_path)
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_preview_label.config(image=photo, text='')
            self.image_preview_label.image = photo

            image2 = Image.open(file_path)
            image2.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo2 = ImageTk.PhotoImage(image2)
            self.inference_preview_label.config(image=photo2, text='')
            self.inference_preview_label.image = photo2

            filename = os.path.basename(file_path)
            self.filename_label.config(text=f'已选择: {filename}')
            self.update_status(f'已加载图片: {filename}')

    def select_folder(self):
        """选择文件夹"""
        folder_path = filedialog.askdirectory(title="选择图片文件夹")

        if folder_path:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            image_files = []

            for ext in extensions:
                image_files.extend(Path(folder_path).glob(f'*{ext}'))
                image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))

            self.batch_image_paths = [str(f) for f in image_files]
            self.current_mode = 'batch'

            count = len(self.batch_image_paths)
            self.batch_info_label.config(
                text=f'已选择 {count} 张图片\n文件夹: {os.path.basename(folder_path)}'
            )
            self.update_status(f'已选择文件夹: {count} 张图片')

    def select_multiple_files(self):
        """选择多个文件"""
        file_paths = filedialog.askopenfilenames(
            title="选择多张图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if file_paths:
            self.batch_image_paths = list(file_paths)
            self.current_mode = 'batch'

            count = len(self.batch_image_paths)
            self.batch_info_label.config(text=f'已选择 {count} 张图片')
            self.update_status(f'已选择 {count} 张图片')

    def start_batch_processing(self):
        """开始批量处理"""
        if not self.batch_image_paths:
            messagebox.showwarning('警告', '请先选择图片!')
            self.notebook.select(1)
            return

        selected_models = [
            model_id for model_id, var in self.model_vars.items()
            if var.get()
        ]

        if not selected_models:
            messagebox.showwarning('警告', '请至少选择一个模型!')
            return

        self.progress.pack(pady=10)
        self.progress['value'] = 0
        self.notebook.select(4 if IPCameraCapture else 3)

        processor = BatchProcessor(
            self.batch_image_paths,
            selected_models,
            self.dataset_type
        )

        processor.set_callbacks(
            self.on_batch_progress,
            self.on_batch_item_finished,
            self.on_batch_finished,
            self.on_batch_error
        )

        thread = threading.Thread(target=processor.run, daemon=True)
        thread.start()

    def on_batch_progress(self, value: int, message: str):
        """批量处理进度"""
        self.progress['value'] = value
        self.update_status(message)
        self.root.update_idletasks()

    def on_batch_item_finished(self, batch_result: BatchResult):
        """单个批量项完成"""
        filename = os.path.basename(batch_result.image_path)
        self.update_status(f'✅ 完成: {filename}')

    def on_batch_finished(self, batch_results: List[BatchResult]):
        """批量处理完成"""
        self.batch_results = batch_results
        self.progress.pack_forget()
        self.results_manager.display_batch_results(batch_results)
        self.update_status('✅ 批量处理完成!')
        messagebox.showinfo('完成', f'成功处理 {len(batch_results)} 张图片!')

    def on_batch_error(self, error_msg: str):
        """批量处理错误"""
        self.progress.pack_forget()
        messagebox.showerror('错误', error_msg)
        self.update_status('❌ 批量处理失败')

    # ==================== 单图推断 ====================

    def start_single_inference(self):
        """开始单图推断"""
        if not self.uploaded_image_path:
            messagebox.showwarning('警告', '请先上传图片或使用摄像头捕获!')
            return

        selected_models = [
            model_id for model_id, var in self.model_vars.items()
            if var.get()
        ]

        if not selected_models:
            messagebox.showwarning('警告', '请至少选择一个模型!')
            return

        self.progress.pack(pady=10)
        self.progress['value'] = 0
        tab_index = 4 if IPCameraCapture else 3
        self.notebook.select(tab_index)

        worker = InferenceWorker(
            self.uploaded_image_path,
            selected_models,
            self.dataset_type
        )

        worker.set_callbacks(
            self.on_progress,
            self.on_finished,
            self.on_error
        )

        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()

    def on_progress(self, value: int, message: str):
        """更新进度"""
        self.progress['value'] = value
        self.update_status(message)
        self.root.update_idletasks()

    def on_finished(self, results: Dict[str, InferenceResult]):
        """推断完成"""
        self.inference_results = results
        self.progress.pack_forget()

        avg_time = np.mean([r.inference_time for r in results.values()])

        self.results_manager.display_single_result(self.uploaded_image_path, results)
        self.update_status(f'✅ 推断完成! 平均处理时间: {avg_time:.2f}ms')
        messagebox.showinfo('完成', f'所有模型推断完成!\n平均处理时间: {avg_time:.2f}ms')

    def on_error(self, error_msg: str):
        """推断错误"""
        self.progress.pack_forget()
        messagebox.showerror('错误', error_msg)
        self.update_status('❌ 推断失败')

    # ==================== 数据集切换 ====================

    def on_dataset_changed(self, event):
        """数据集切换"""
        self.dataset_type = self.dataset_var.get()
        dataset_name = '橡胶木 (6类)' if self.dataset_type == 'rubber' else '松木 (4类)'
        self.update_status(f'🔄 切换到: {dataset_name}')

    # ==================== 导出报告 ====================

    def export_report(self):
        """导出报告"""
        if not self.inference_results and not self.batch_results:
            messagebox.showwarning('警告', '暂无结果可导出!')
            return

        save_path = filedialog.asksaveasfilename(
            title="保存报告",
            defaultextension=".txt",
            filetypes=[
                ("文本文件", "*.txt"),
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("木材缺陷检测报告\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据集类型: {self.dataset_var.get()}\n")
                f.write(f"处理模式: {self.current_mode}\n\n")

                if self.current_mode in ['single', 'camera', 'ip_camera']:
                    f.write(f"输入图片: {os.path.basename(self.uploaded_image_path)}\n\n")
                    f.write("模型性能对比:\n")
                    f.write("-" * 70 + "\n")

                    times = []
                    for model_id, result in self.inference_results.items():
                        model_name = self.available_models[model_id]['name']
                        f.write(f"\n{model_name}:\n")
                        f.write(f"  mIoU: {result.metrics['mIoU']:.4f}\n")
                        f.write(f"  mAcc: {result.metrics['mAcc']:.4f}\n")
                        f.write(f"  F1: {result.metrics['F1']:.4f}\n")
                        f.write(f"  推断时间: {result.inference_time:.2f}ms\n")
                        times.append(result.inference_time)

                    avg_time = np.mean(times)
                    f.write(f"\n平均处理时间: {avg_time:.2f}ms\n")

                elif self.current_mode == 'batch':
                    f.write(f"处理图片数量: {len(self.batch_results)}\n\n")
                    f.write("批量处理统计:\n")
                    f.write("-" * 70 + "\n")

                    model_stats = {}
                    for batch_result in self.batch_results:
                        for model_id, result in batch_result.results.items():
                            if model_id not in model_stats:
                                model_stats[model_id] = {
                                    'mIoU': [],
                                    'mAcc': [],
                                    'F1': [],
                                    'time': []
                                }
                            model_stats[model_id]['mIoU'].append(result.metrics['mIoU'])
                            model_stats[model_id]['mAcc'].append(result.metrics['mAcc'])
                            model_stats[model_id]['F1'].append(result.metrics['F1'])
                            model_stats[model_id]['time'].append(result.inference_time)

                    for model_id, stats in model_stats.items():
                        model_name = self.available_models[model_id]['name']
                        f.write(f"\n{model_name} (平均值):\n")
                        f.write(f"  mIoU: {np.mean(stats['mIoU']):.4f}\n")
                        f.write(f"  mAcc: {np.mean(stats['mAcc']):.4f}\n")
                        f.write(f"  F1: {np.mean(stats['F1']):.4f}\n")
                        f.write(f"  平均推断时间: {np.mean(stats['time']):.2f}ms\n")

            messagebox.showinfo('成功', f'报告已保存至:\n{save_path}')
            self.update_status(f'💾 报告已导出: {os.path.basename(save_path)}')

    # ==================== 清除所有 ====================

    def clear_all(self):
        """清除所有数据"""
        result = messagebox.askyesno('确认', '确定要清除所有数据吗?')

        if result:
            # 停止摄像头
            if self.local_camera_running:
                self.stop_local_camera()
            if self.ip_camera_running:
                self.stop_ip_camera()

            # 清除数据
            self.uploaded_image_path = None
            self.batch_image_paths = []
            self.inference_results = {}
            self.batch_results = []

            self.local_realtime_results = []

            # 清除UI
            self.image_preview_label.config(
                image='',
                text='点击下方按钮上传图片\n支持 JPG, PNG 格式\n推荐尺寸 512×512'
            )
            self.inference_preview_label.config(image='', text='暂无图片')
            self.filename_label.config(text='')
            self.batch_info_label.config(text='')

            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            for widget in self.vis_frame.winfo_children():
                widget.destroy()

            self.notebook.select(0)
            self.update_status('🗑️ 已清除所有数据')

    # ==================== 工具方法 ====================

    def update_status(self, message: str):
        """更新状态栏"""
        self.statusbar.config(text=message)
        self.root.update_idletasks()


# ==================== 主函数 ====================

def main():
    """主函数"""
    root = tk.Tk()
    app = WoodDefectGUI(root)

    # 优雅退出
    def on_closing():
        if app.local_camera_running:
            app.stop_local_camera()
        if IPCameraCapture and app.ip_camera_running:
            app.stop_ip_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()