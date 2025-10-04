from __future__ import annotations

import argparse
import os
from typing import Optional
import cv2
from tqdm import tqdm

from src.detector.yolo_face import YoloFaceDetector
from src.embedding.insight_embedder import InsightFaceEmbedder
from src.utils.db import FaceDatabase
from src.utils.vis import draw_detections


def process_image(path: str, detector: YoloFaceDetector, embedder: InsightFaceEmbedder, db: FaceDatabase, save_path: Optional[str] = None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    dets = detector.detect(img)
    crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in dets]
    embs = embedder.get_embeddings(crops)
    labels, scores = db.search(embs)
    label_texts = [f"{l} {s:.2f}" for l, s in zip(labels, scores)]
    vis = draw_detections(img, dets, label_texts)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
    else:
        cv2.imshow("result", vis)
        cv2.waitKey(0)
    return labels, scores


def process_video(source: str | int, detector: YoloFaceDetector, embedder: InsightFaceEmbedder, db: FaceDatabase, output: Optional[str] = None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in dets]
            embs = embedder.get_embeddings(crops)
            labels, scores = db.search(embs)
            label_texts = [f"{l} {s:.2f}" for l, s in zip(labels, scores)]
            vis = draw_detections(frame, dets, label_texts)

            if writer is not None:
                writer.write(vis)
            else:
                cv2.imshow("result", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Recognize faces using YOLO + InsightFace")
    parser.add_argument("--image", type=str, help="Path to image to process")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")
    parser.add_argument("--db", type=str, default="data/faces_db.json", help="Path to face database JSON")
    parser.add_argument("--weights", type=str, default="yolov8n-face.pt", help="YOLO face weights path")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence for detection")
    parser.add_argument("--output", type=str, default=None, help="Output image/video path")
    args = parser.parse_args()

    detector = YoloFaceDetector(model_path=args.weights, conf_threshold=args.min_conf)
    embedder = InsightFaceEmbedder()
    db = FaceDatabase(db_path=args.db)

    if args.image:
        process_image(args.image, detector, embedder, db, save_path=args.output)
    elif args.video:
        process_video(args.video, detector, embedder, db, output=args.output)
    elif args.webcam:
        process_video(0, detector, embedder, db, output=args.output)
    else:
        parser.error("Please provide --image or --video or --webcam")


if __name__ == "__main__":
    main()
