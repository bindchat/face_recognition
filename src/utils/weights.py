from __future__ import annotations

import os
import urllib.request
from typing import Optional


YOLO_FACE_URL = "https://github.com/derronqi/yolov5-face/releases/download/v1.0/yolov8n-face.pt"


def download_yolo_face_weights(dest_path: str = "yolov8n-face.pt", url: Optional[str] = None) -> str:
    src_url = url or YOLO_FACE_URL
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        return dest_path
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading YOLO face weights from {src_url} ...")
    urllib.request.urlretrieve(src_url, dest_path)
    print(f"Saved weights to {dest_path}")
    return dest_path
