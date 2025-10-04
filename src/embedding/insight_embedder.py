from __future__ import annotations

from typing import List, Optional
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class InsightFaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", providers: Optional[list[str]] = None):
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embeddings(self, face_images_bgr: List[np.ndarray]) -> np.ndarray:
        if not face_images_bgr:
            return np.empty((0, 512), dtype=np.float32)
        embeddings: List[np.ndarray] = []
        for img in face_images_bgr:
            if img is None or img.size == 0:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)
            if not faces:
                continue
            # Use the largest detected face in the crop
            faces.sort(key=lambda f: f.bbox[2] * f.bbox[3] - f.bbox[0] * f.bbox[1], reverse=True)
            emb = faces[0].normed_embedding.astype(np.float32)
            embeddings.append(emb)
        if not embeddings:
            return np.empty((0, 512), dtype=np.float32)
        return np.vstack(embeddings)
