"""
一个小脚本：运行它就会把 YOLO 人脸模型的权重文件下载到本地。
"""

import os
import sys

# 保证从项目根目录运行或直接执行本脚本时都能找到 `src` 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.weights import download_yolo_face_weights

if __name__ == "__main__":
    # 直接调用工具函数下载权重
    download_yolo_face_weights()
