from __future__ import annotations

"""
这个文件帮我们把 YOLO 人脸模型的“重量文件”（权重）下载到本地。
可以把权重理解成“模型学到的知识”，用它就能更聪明地找脸。
"""

import os
import urllib.request
from typing import Optional


YOLO_FACE_URL = "https://github.com/derronqi/yolov5-face/releases/download/v1.0/yolov8n-face.pt"


def download_yolo_face_weights(dest_path: str = "yolov8n-face.pt", url: Optional[str] = None) -> str:
    """如果本地没有权重文件，就从网络下载一个；如果有了就直接用。"""
    src_url = url or YOLO_FACE_URL
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        # 文件已经存在而且看起来不小，认为下载过了
        return dest_path
    # 确保保存文件的文件夹存在
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading YOLO face weights from {src_url} ...")
    urllib.request.urlretrieve(src_url, dest_path)
    print(f"Saved weights to {dest_path}")
    return dest_path
