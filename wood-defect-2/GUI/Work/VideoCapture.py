"""
视频处理模块
支持从视频文件中读取帧并进行检测
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
from typing import Optional, Callable, Dict, List
from pathlib import Path


class VideoCapture:
    """视频处理类"""

    def __init__(self, video_path: str = None):
        """
        初始化视频处理器

        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # 视频信息
        self.total_frames = 0
        self.current_frame_pos = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.duration = 0  # 秒

        # 回调函数
        self.frame_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None

        # 播放控制
        self.playback_speed = 1.0  # 播放速度倍率

        print(f" 视频处理器初始化: {video_path}")

    def set_callbacks(self, frame_cb: Callable = None, error_cb: Callable = None,
                      progress_cb: Callable = None):
        """设置回调函数"""
        self.frame_callback = frame_cb
        self.error_callback = error_cb
        self.progress_callback = progress_cb

    def load_video(self, video_path: str) -> bool:
        """
        加载视频文件

        Args:
            video_path: 视频文件路径

        Returns:
            是否加载成功
        """
        try:
            self.video_path = video_path

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")

            # 打开视频
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                raise Exception("无法打开视频文件")

            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0

            print(f" 视频加载成功:")
            print(f"   分辨率: {self.width}x{self.height}")
            print(f"   帧率: {self.fps:.2f} fps")
            print(f"   总帧数: {self.total_frames}")
            print(f"   时长: {self.duration:.2f}秒")

            return True

        except Exception as e:
            error_msg = f"视频加载失败: {str(e)}"
            print(f" {error_msg}")
            if self.error_callback:
                self.error_callback(error_msg)
            return False

    def start(self) -> bool:
        """开始播放视频"""
        if not self.cap or not self.cap.isOpened():
            print(" 请先加载视频文件")
            return False

        if self.is_running:
            print("️  视频已在播放")
            return True

        try:
            self.is_running = True
            self.is_paused = False

            # 启动播放线程
            thread = threading.Thread(target=self._playback_loop, daemon=True)
            thread.start()

            print("️  开始播放视频")
            return True

        except Exception as e:
            error_msg = f"视频播放启动失败: {str(e)}"
            print(f" {error_msg}")
            if self.error_callback:
                self.error_callback(error_msg)
            return False

    def stop(self):
        """停止播放"""
        print("️  停止播放视频")
        self.is_running = False
        self.is_paused = False

        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        """暂停播放"""
        self.is_paused = True
        print("️  暂停播放")

    def resume(self):
        """恢复播放"""
        self.is_paused = False
        print("️  恢复播放")

    def seek(self, frame_number: int):
        """
        跳转到指定帧

        Args:
            frame_number: 目标帧号
        """
        if self.cap and self.cap.isOpened():
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_pos = frame_number
            print(f"️  跳转到第 {frame_number} 帧")

    def set_playback_speed(self, speed: float):
        """
        设置播放速度

        Args:
            speed: 速度倍率 (0.5 = 慢放, 1.0 = 正常, 2.0 = 快放)
        """
        self.playback_speed = max(0.1, min(speed, 5.0))
        print(f" 播放速度设置为 {self.playback_speed}x")

    def _playback_loop(self):
        """播放循环(在独立线程中运行)"""
        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033

        while self.is_running and self.cap and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # 更新当前帧位置
                    self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                    # 转换BGR到RGB
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

                    # 进度回调
                    if self.progress_callback:
                        progress = (self.current_frame_pos / self.total_frames) * 100
                        self.progress_callback(self.current_frame_pos, self.total_frames, progress)

                    # 控制播放速度
                    adjusted_delay = frame_delay / self.playback_speed
                    time.sleep(adjusted_delay)

                else:
                    # 视频播放完毕
                    print("✅ 视频播放完毕")
                    self.is_running = False
                    break

            except Exception as e:
                print(f" 播放出错: {str(e)}")
                time.sleep(0.1)

        print("播放循环结束")

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧(从队列)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧(直接从缓存)"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def capture_current_frame(self) -> Optional[np.ndarray]:
        """捕获当前帧"""
        return self.get_latest_frame()

    def extract_frames(self, interval: int = 1, max_frames: int = None) -> List[np.ndarray]:
        """
        提取视频帧

        Args:
            interval: 帧间隔(每隔多少帧提取一帧)
            max_frames: 最大提取帧数

        Returns:
            提取的帧列表
        """
        if not self.cap or not self.cap.isOpened():
            print("❌ 请先加载视频文件")
            return []

        frames = []
        frame_count = 0
        extracted_count = 0

        # 重置到视频开始
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print(f" 开始提取帧 (间隔: {interval}, 最大数: {max_frames or '无限制'})")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            if frame_count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        print(f" 提取完成: 共 {len(frames)} 帧")

        # 重置到开始
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return frames

    def get_info(self) -> Dict:
        """获取视频信息"""
        return {
            'path': self.video_path,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'current_frame': self.current_frame_pos,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'playback_speed': self.playback_speed
        }

    def release(self):
        """释放资源"""
        self.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        print("🔄 视频资源已释放")


# 使用示例
if __name__ == '__main__':
    # 创建视频处理器
    video = VideoCapture()

    # 加载视频
    if video.load_video('test_video.mp4'):
        # 提取帧
        frames = video.extract_frames(interval=30, max_frames=10)
        print(f"提取了 {len(frames)} 帧")

        # 释放资源
        video.release()