from __future__ import annotations

import argparse
import os
from pathlib import Path
import cv2
from tqdm import tqdm

from src.detector.yolo_face import YoloFaceDetector
from src.embedding.insight_embedder import InsightFaceEmbedder
from src.utils.db import FaceDatabase


def iter_images(root: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                yield os.path.join(dirpath, fn)


def main():
    parser = argparse.ArgumentParser(description="Enroll faces from a folder into database")
    parser.add_argument("data_dir", type=str, help="Folder with subfolders per identity, containing face images")
    parser.add_argument("--db", type=str, default="data/faces_db.json", help="Path to face database JSON")
    parser.add_argument("--weights", type=str, default="yolov8n-face.pt", help="YOLO face weights path")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence for detection")
    args = parser.parse_args()

    detector = YoloFaceDetector(model_path=args.weights, conf_threshold=args.min_conf)
    embedder = InsightFaceEmbedder()
    db = FaceDatabase(db_path=args.db)

    id_to_crops = {}
    for img_path in tqdm(list(iter_images(args.data_dir)), desc="Scanning images"):
        label = Path(img_path).parent.name
        img = cv2.imread(img_path)
        if img is None:
            continue
        crops = detector.crop_faces(img)
        if not crops:
            continue
        if label not in id_to_crops:
            id_to_crops[label] = []
        id_to_crops[label].extend(crops)

    for name, crops in id_to_crops.items():
        embs = embedder.get_embeddings(crops)
        if embs.size:
            db.add(name, embs)
            print(f"Enrolled {name}: {embs.shape[0]} embeddings")


if __name__ == "__main__":
    main()
