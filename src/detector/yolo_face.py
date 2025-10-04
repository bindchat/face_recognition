from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2
from ultralytics import YOLO


class YoloFaceDetector:
    def __init__(self, model_path: str = "yolov8n-face.pt", conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        results = self.model.predict(source=image_bgr, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
        detections: List[Tuple[np.ndarray, float]] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                detections.append((np.array([x1, y1, x2, y2], dtype=int), conf))
        return detections

    def crop_faces(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        crops = []
        for (x1, y1, x2, y2), _ in self.detect(image_bgr):
            crops.append(image_bgr[y1:y2, x1:x2].copy())
        return crops
