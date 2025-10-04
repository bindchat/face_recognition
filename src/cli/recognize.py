from __future__ import annotations

"""
这个文件做“认人”的工作：
- 先在图片或视频里找到人脸。
- 再把人脸变成“数字指纹”。
- 拿这些指纹去数据库里比一比，看看最像哪一个人。
- 最后在画面上画出方框和名字，给你看结果。
"""

import argparse
import os
from typing import Optional
import cv2
from tqdm import tqdm

from src.detector.yolo_face import YoloFaceDetector
from src.embedding.insight_embedder import InsightFaceEmbedder
from src.utils.db import FaceDatabase
from src.utils.vis import draw_detections


def process_image(
    path: str,
    detector: YoloFaceDetector,
    embedder: InsightFaceEmbedder,
    db: FaceDatabase,
    save_path: Optional[str] = None,
):
    """处理一张图片，找人、认人，并把结果画出来。

    返回：
    - labels: 识别出的名字列表（和检测到的人脸一一对应）
    - scores: 每个名字的相似度分数（越高越像）
    """

    # 读入图片
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    # 1) 找到图片里的人脸方框
    dets = detector.detect(img)

    # 2) 把每个方框里的脸剪下来
    crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in dets]

    # 3) 把脸变成“数字指纹”（向量）
    embs = embedder.get_embeddings(crops)

    # 4) 去小数据库里找最像的名字
    labels, scores = db.search(embs)

    # 5) 画结果：绿色方框 + 名字与分数
    label_texts = [f"{l} {s:.2f}" for l, s in zip(labels, scores)]
    vis = draw_detections(img, dets, label_texts)

    # 6) 保存到文件，或者弹窗显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
    else:
        cv2.imshow("result", vis)
        cv2.waitKey(0)

    return labels, scores


def process_video(
    source: str | int,
    detector: YoloFaceDetector,
    embedder: InsightFaceEmbedder,
    db: FaceDatabase,
    output: Optional[str] = None,
):
    """一边读视频/摄像头，一边认人并把结果画到每一帧上。

    - source 可以是视频文件路径，也可以是数字 0（表示电脑摄像头）。
    - 如果给了 output，就把结果保存成视频文件。
    """

    # 打开视频或摄像头
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    if output:
        # 为了把画好结果的每一帧写进视频文件，需要先准备写视频的工具
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    try:
        while True:
            # 读一帧图像
            ret, frame = cap.read()
            if not ret:
                break

            # 找人脸 → 剪脸 → 算向量 → 去数据库查名字 → 画结果
            dets = detector.detect(frame)
            crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in dets]
            embs = embedder.get_embeddings(crops)
            labels, scores = db.search(embs)
            label_texts = [f"{l} {s:.2f}" for l, s in zip(labels, scores)]
            vis = draw_detections(frame, dets, label_texts)

            # 写入视频或直接显示
            if writer is not None:
                writer.write(vis)
            else:
                cv2.imshow("result", vis)
                # 按下 ESC 键（27）就退出
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    """命令行入口：根据你的选择处理一张图片、一个视频，或打开摄像头。"""

    parser = argparse.ArgumentParser(description="Recognize faces using YOLO + InsightFace")
    parser.add_argument("--image", type=str, help="Path to image to process")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")
    parser.add_argument("--db", type=str, default="data/faces_db.json", help="Path to face database JSON")
    parser.add_argument("--weights", type=str, default="yolov8n-face.pt", help="YOLO face weights path")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence for detection")
    parser.add_argument("--output", type=str, default=None, help="Output image/video path")
    args = parser.parse_args()

    # 准备三位“帮手”
    detector = YoloFaceDetector(model_path=args.weights, conf_threshold=args.min_conf)
    embedder = InsightFaceEmbedder()
    db = FaceDatabase(db_path=args.db)

    # 根据命令行选项决定怎么运行
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
