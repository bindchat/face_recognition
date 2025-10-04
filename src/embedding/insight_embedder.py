from __future__ import annotations

"""
这个文件负责把“脸的图片”变成“数字指纹”（一个长度为 512 的向量）。
你可以把它想象成：
- 输入：一张张小小的脸图。
- 输出：每张脸得到 512 个数字，这些数字能代表这个人的样子。
以后我们就用这些数字来对比两张脸是不是同一个人。
"""

from typing import List, Optional
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class InsightFaceEmbedder:
    """把脸图变成“向量”的小帮手。"""

    def __init__(self, model_name: str = "buffalo_l", providers: Optional[list[str]] = None):
        # 创建 insightface 的分析器。它能在脸上提取“特征”。
        self.app = FaceAnalysis(name=model_name, providers=providers)
        # 准备工作：选用 GPU/CPU（ctx_id=0 代表用第 0 个设备），并设置输入大小
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embeddings(self, face_images_bgr: List[np.ndarray]) -> np.ndarray:
        """把很多张脸图，依次变成“512 维向量”。

        参数：
        - face_images_bgr: 裁剪好的人脸小图片，颜色顺序是 BGR。

        返回：
        - 形状是 (N, 512) 的数组，每一行是一个脸的“数字指纹”。
        """

        if not face_images_bgr:
            # 没有输入时，返回一个空的 (0, 512) 数组，方便后续代码处理
            return np.empty((0, 512), dtype=np.float32)

        embeddings: List[np.ndarray] = []
        for img in face_images_bgr:
            if img is None or img.size == 0:
                # 图片坏了或空的，就跳过
                continue

            # insightface 习惯用 RGB，所以把 BGR 转成 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 让模型在这张小图里再找一次脸（更准确地定位）
            faces = self.app.get(img_rgb)
            if not faces:
                # 这里可能是脸太小或太模糊，就跳过
                continue

            # 如果找到多张脸，挑最大的那一张（通常是我们要的）
            faces.sort(
                key=lambda f: f.bbox[2] * f.bbox[3] - f.bbox[0] * f.bbox[1],
                reverse=True,
            )

            # normed_embedding 已经是归一化过的 512 维向量
            emb = faces[0].normed_embedding.astype(np.float32)
            embeddings.append(emb)

        if not embeddings:
            # 如果一张也没成功，就返回空数组
            return np.empty((0, 512), dtype=np.float32)

        # 把很多个 (512,) 叠在一起，得到 (N, 512)
        return np.vstack(embeddings)
