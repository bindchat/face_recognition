"""
一个小脚本：运行它就会把 YOLO 人脸模型的权重文件下载到本地。
"""

from src.utils.weights import download_yolo_face_weights

if __name__ == "__main__":
    # 直接调用工具函数下载权重
    download_yolo_face_weights()
