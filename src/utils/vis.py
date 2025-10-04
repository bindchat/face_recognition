from __future__ import annotations

"""
这个文件帮助我们把“识别结果”画在图片上：
- 给人脸画绿色方框；
- 在方框上写出名字或分数；
- 还提供一个安全读取图片的小函数。
"""

from typing import List, Tuple
import numpy as np
import cv2


def draw_detections(
    image_bgr: np.ndarray,
    dets: List[Tuple[np.ndarray, float]],
    labels: List[str] | None = None,
) -> np.ndarray:
    """在图片上画出每个检测到的人脸方框，并写上标签。"""
    output = image_bgr.copy()
    for i, ((x1, y1, x2, y2), conf) in enumerate(dets):
        # 如果有提供名字就用名字，否则就用分数作为文字
        label = labels[i] if labels and i < len(labels) else f"{conf:.2f}"
        # 画绿色方框，线条宽度为 2
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 在方框上方写文字
        cv2.putText(output, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return output


def load_image(path: str) -> np.ndarray:
    """从磁盘读取一张图片，如果失败就告诉你文件找不到。"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {path}")
    return img
