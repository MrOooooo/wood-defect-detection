import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from GUI.Data.InterfenceResult import InferenceResult


@dataclass
class BatchResult:
    """批量处理结果"""
    image_path: str
    results: Dict[str, InferenceResult]
    avg_time: float  # 平均处理时间