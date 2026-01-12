"""
IP 摄像头采集模块
支持 RTSP、HTTP、ONVIF 等多种协议
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, Dict, List
from urllib.parse import urlparse
import requests


class IPCameraCapture:
    """IP 摄像头采集类"""

    # 常见 IP 摄像头 URL 模板
    URL_TEMPLATES = {
        'hikvision_rtsp': 'rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}',
        'dahua_rtsp': 'rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0',
        'generic_rtsp': 'rtsp://{user}:{password}@{ip}:{port}/stream{channel}',
        'http_mjpeg': 'http://{ip}:{port}/video',
        'onvif': 'rtsp://{user}:{password}@{ip}:{port}/onvif1',
    }

    def __init__(self,
                 camera_url: str = None,
                 camera_type: str = 'generic_rtsp',
                 ip: str = '192.168.1.64',
                 port: int = 554,
                 user: str = 'admin',
                 password: str = 'admin',
                 channel: int = 1,
                 buffer_size: int = 2):
        """
        初始化 IP 摄像头

        Args:
            camera_url: 完整的摄像头 URL（如果提供，其他参数将被忽略）
            camera_type: 摄像头类型（从 URL_TEMPLATES 中选择）
            ip: 摄像头 IP 地址
            port: 端口号
            user: 用户名
            password: 密码
            channel: 通道号
            buffer_size: 缓冲区大小
        """
        if camera_url:
            self.camera_url = camera_url
        else:
            # 根据模板构建 URL
            template = self.URL_TEMPLATES.get(camera_type, self.URL_TEMPLATES['generic_rtsp'])
            self.camera_url = template.format(
                user=user,
                password=password,
                ip=ip,
                port=port,
                channel=channel
            )

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # 回调函数
        self.frame_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None

        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.connection_attempts = 0
        self.max_reconnect_attempts = 5

        # 录制相关
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None

        print(f"📹 IP摄像头配置: {self._mask_password(self.camera_url)}")

    def _mask_password(self, url: str) -> str:
        """隐藏 URL 中的密码"""
        if '@' in url and '//' in url:
            parts = url.split('//')
            if len(parts) > 1 and '@' in parts[1]:
                auth_part = parts[1].split('@')[0]
                if ':' in auth_part:
                    user = auth_part.split(':')[0]
                    return url.replace(auth_part, f"{user}:****")
        return url

    def set_callbacks(self, frame_cb: Callable = None, error_cb: Callable = None):
        """设置回调函数"""
        self.frame_callback = frame_cb
        self.error_callback = error_cb

    def test_connection(self) -> bool:
        """测试摄像头连接"""
        print("🔍 测试 IP 摄像头连接...")
        try:
            test_cap = cv2.VideoCapture(self.camera_url)

            # 设置超时
            test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 尝试读取一帧
            for i in range(10):  # 最多尝试10次
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✅ 连接成功! 分辨率: {width}x{height}")
                    test_cap.release()
                    return True
                time.sleep(0.1)

            test_cap.release()
            print("❌ 无法读取视频帧")
            return False

        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            return False

    def start(self) -> bool:
        """启动摄像头采集"""
        if self.is_running:
            print("⚠️  摄像头已在运行")
            return True

        try:
            print("🚀 启动 IP 摄像头...")

            # 打开视频流
            self.cap = cv2.VideoCapture(self.camera_url)

            if not self.cap.isOpened():
                raise Exception("无法打开 IP 摄像头")

            # 配置 OpenCV 参数
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化延迟

            # 获取视频信息
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"📐 视频流参数: {width}x{height} @ {fps}fps")

            self.is_running = True
            self.connection_attempts = 0

            # 启动采集线程
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            thread.start()

            # 启动 FPS 计算线程
            fps_thread = threading.Thread(target=self._fps_calculator, daemon=True)
            fps_thread.start()

            print("✅ IP 摄像头启动成功!")
            return True

        except Exception as e:
            error_msg = f"IP 摄像头启动失败: {str(e)}"
            print(f"❌ {error_msg}")
            if self.error_callback:
                self.error_callback(error_msg)
            return False

    def stop(self):
        """停止摄像头采集"""
        print("🛑 停止 IP 摄像头...")
        self.is_running = False

        # 停止录制
        if self.is_recording:
            self.stop_recording()

        # 释放资源
        if self.cap:
            self.cap.release()
            self.cap = None

        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        print("✅ IP 摄像头已停止")

    def _capture_loop(self):
        """采集循环（在独立线程中运行）"""
        consecutive_failures = 0
        max_failures = 30  # 连续失败30次后尝试重连

        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(1)
                        continue

                ret, frame = self.cap.read()

                if ret and frame is not None:
                    consecutive_failures = 0

                    # 转换 BGR 到 RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 更新最新帧
                    with self.frame_lock:
                        self.latest_frame = frame_rgb.copy()

                    # 更新队列
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break

                    try:
                        self.frame_queue.put_nowait(frame_rgb)
                    except queue.Full:
                        pass

                    # 调用回调
                    if self.frame_callback:
                        self.frame_callback(frame_rgb)

                    # 更新帧计数
                    self.frame_count += 1

                    # 录制
                    if self.is_recording and self.video_writer:
                        self.video_writer.write(frame)  # 注意这里用 BGR

                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"⚠️  连续失败 {consecutive_failures} 次，尝试重连...")
                        if not self._reconnect():
                            time.sleep(1)
                        consecutive_failures = 0
                    time.sleep(0.01)

            except Exception as e:
                print(f"❌ 采集出错: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    if not self._reconnect():
                        time.sleep(1)
                    consecutive_failures = 0
                time.sleep(0.1)

        print("🔚 采集循环结束")

    def _reconnect(self) -> bool:
        """尝试重新连接"""
        if self.connection_attempts >= self.max_reconnect_attempts:
            error_msg = f"重连失败次数过多 ({self.max_reconnect_attempts})"
            print(f"❌ {error_msg}")
            if self.error_callback:
                self.error_callback(error_msg)
            return False

        self.connection_attempts += 1
        print(f"🔄 尝试重新连接 (第 {self.connection_attempts}/{self.max_reconnect_attempts} 次)...")

        if self.cap:
            self.cap.release()

        time.sleep(2)

        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            if self.cap.isOpened():
                print("✅ 重连成功!")
                self.connection_attempts = 0
                return True
            else:
                print("❌ 重连失败")
                return False
        except Exception as e:
            print(f"❌ 重连异常: {str(e)}")
            return False

    def _fps_calculator(self):
        """FPS 计算线程"""
        while self.is_running:
            time.sleep(1)
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧（从队列）"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧（直接从缓存）"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """捕获快照"""
        return self.get_latest_frame()

    def get_fps(self) -> float:
        """获取当前 FPS"""
        return self.fps

    def start_recording(self, output_path: str, fps: int = 25) -> bool:
        """
        开始录制视频

        Args:
            output_path: 输出文件路径
            fps: 帧率
        """
        if self.is_recording:
            print("⚠️  已在录制中")
            return False

        try:
            frame = self.get_latest_frame()
            if frame is None:
                print("❌ 无法获取视频帧")
                return False

            height, width = frame.shape[:2]

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not self.video_writer.isOpened():
                print("❌ 无法创建视频文件")
                return False

            self.is_recording = True
            print(f"🎥 开始录制: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 录制启动失败: {str(e)}")
            return False

    def stop_recording(self):
        """停止录制视频"""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print("⏹️  录制已停止")

    def get_info(self) -> Dict:
        """获取摄像头信息"""
        info = {
            'url': self._mask_password(self.camera_url),
            'is_running': self.is_running,
            'fps': self.fps,
            'is_recording': self.is_recording,
        }

        if self.cap and self.cap.isOpened():
            info.update({
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            })

        return info


class MultiIPCameraManager:
    """多路 IP 摄像头管理器"""

    def __init__(self):
        self.cameras: Dict[str, IPCameraCapture] = {}

    def add_camera(self, name: str, camera: IPCameraCapture):
        """添加摄像头"""
        self.cameras[name] = camera
        print(f"✅ 添加摄像头: {name}")

    def remove_camera(self, name: str):
        """移除摄像头"""
        if name in self.cameras:
            self.cameras[name].stop()
            del self.cameras[name]
            print(f"🗑️  移除摄像头: {name}")

    def start_all(self):
        """启动所有摄像头"""
        for name, camera in self.cameras.items():
            print(f"启动 {name}...")
            camera.start()

    def stop_all(self):
        """停止所有摄像头"""
        for name, camera in self.cameras.items():
            print(f"停止 {name}...")
            camera.stop()

    def get_camera(self, name: str) -> Optional[IPCameraCapture]:
        """获取指定摄像头"""
        return self.cameras.get(name)

    def get_all_cameras(self) -> Dict[str, IPCameraCapture]:
        """获取所有摄像头"""
        return self.cameras


# 使用示例
if __name__ == '__main__':
    # 示例 1: 海康威视 IP 摄像头
    camera1 = IPCameraCapture(
        camera_type='hikvision_rtsp',
        ip='192.168.1.64',
        port=554,
        user='admin',
        password='your_password',
        channel=1
    )

    # 示例 2: 使用完整 URL
    camera2 = IPCameraCapture(
        camera_url='rtsp://admin:password@192.168.1.65:554/stream1'
    )

    # 测试连接
    if camera1.test_connection():
        print("摄像头连接测试成功!")

        # 启动采集
        camera1.start()

        # 等待几秒
        time.sleep(5)

        # 捕获快照
        snapshot = camera1.capture_snapshot()
        if snapshot is not None:
            print(f"捕获快照: {snapshot.shape}")

        # 查看 FPS
        print(f"当前 FPS: {camera1.get_fps():.2f}")

        # 停止
        camera1.stop()