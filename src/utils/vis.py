from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2


def draw_detections(image_bgr: np.ndarray, dets: List[Tuple[np.ndarray, float]], labels: List[str] | None = None) -> np.ndarray:
    output = image_bgr.copy()
    for i, ((x1, y1, x2, y2), conf) in enumerate(dets):
        label = labels[i] if labels and i < len(labels) else f"{conf:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return output


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {path}")
    return img
