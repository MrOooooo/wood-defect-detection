
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass



@dataclass
class InferenceResult:
    """推断结果数据结构"""
    model_name: str
    segmentation: np.ndarray  # 叠加后的图像
    pred_mask: np.ndarray  # 预测mask
    metrics: Dict[str, float]  # 评估指标
    class_distribution: List[float]  # 类别分布
    inference_time: float  # 推断时间(ms)