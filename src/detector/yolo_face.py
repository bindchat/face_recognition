from __future__ import annotations

"""
这个文件负责“找脸”，也就是在人像照片里画出脸的方框。
我们用的是 YOLO 模型，它像一个很会找东西的机器人。
"""

from typing import List, Tuple
import numpy as np
import cv2
from ultralytics import YOLO


class YoloFaceDetector:
    """用 YOLO 模型在图片里找人脸的小帮手。"""

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        # 加载训练好的模型权重
        self.model = YOLO(model_path)
        # 置信度阈值：分数太低的结果不要
        self.conf_threshold = float(conf_threshold)
        # IOU 阈值：用来去掉重叠很大的方框
        self.iou_threshold = float(iou_threshold)

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """在一张图片里找到所有人脸的方框。

        返回：一个列表，里面的每个元素是 (方框坐标, 分数)。
        方框坐标是 [x1, y1, x2, y2]，表示左上角和右下角的位置。
        分数越高，说明模型越确定这里有一张脸。
        """

        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        # 用 YOLO 模型做预测
        results = self.model.predict(
            source=image_bgr, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )[0]

        detections: List[Tuple[np.ndarray, float]] = []
        for box in results.boxes:
            # 把预测到的方框坐标和分数取出来
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())

            # 把坐标修正到图片范围内（不能超出边界）
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            # 过滤掉宽或高为负的非法框
            if x2 > x1 and y2 > y1:
                detections.append((np.array([x1, y1, x2, y2], dtype=int), conf))
        return detections

    def crop_faces(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """把 detect 找到的每个方框都剪下来，得到一张张“脸的小图”。"""
        crops = []
        for (x1, y1, x2, y2), _ in self.detect(image_bgr):
            crops.append(image_bgr[y1:y2, x1:x2].copy())
        return crops
