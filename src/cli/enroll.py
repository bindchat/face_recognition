from __future__ import annotations

"""
用很简单的话解释这个文件在做什么：
- 我们从一个大文件夹里找很多照片。
- 用“找脸”的工具在照片里找到人脸的位置。
- 把每一张脸变成一个数字指纹（叫做“向量”）。
- 把“名字”和“数字指纹”一起存进一个小小的数据库里，方便以后认人。
"""

import argparse
import os
from pathlib import Path
import cv2
from tqdm import tqdm

from src.detector.yolo_face import YoloFaceDetector
from src.embedding.insight_embedder import InsightFaceEmbedder
from src.utils.db import FaceDatabase


def iter_images(root: str):
    """在一个文件夹里把所有图片找出来，并一个一个地给出去。

    参数：
    - root: 放照片的大文件夹路径。里面可能有很多子文件夹。

    返回：
    - 这是一个“生成器”：不会一次性返回全部，而是每次给出一个图片的路径。
    """

    # 我们只关心常见的图片格式
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # 走遍 root 文件夹里的所有子文件夹
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # 只挑选后缀名是图片的文件
            if Path(fn).suffix.lower() in exts:
                # 组合成图片的完整路径
                yield os.path.join(dirpath, fn)


def main():
    """把一堆人脸图片“录入”到数据库里。

    简单理解：
    - 输入：一个大文件夹，里面每个子文件夹代表一个人，名字就是子文件夹名。
    - 过程：找到每张照片里的脸，算出每张脸的“数字指纹”。
    - 输出：把“人的名字”和“他的很多数字指纹”保存到数据库文件里。
    """

    # 创建命令行参数解析器：用来读取用户在命令行输入的选项
    parser = argparse.ArgumentParser(description="Enroll faces from a folder into database")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Folder with subfolders per identity, containing face images",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/faces_db.json",
        help="Path to face database JSON",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n-face.pt",
        help="YOLO face weights path",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.5,
        help="Minimum confidence for detection",
    )
    args = parser.parse_args()

    # 准备三个小帮手：找脸的、把脸变成向量的、存数据的
    detector = YoloFaceDetector(model_path=args.weights, conf_threshold=args.min_conf)
    embedder = InsightFaceEmbedder()
    db = FaceDatabase(db_path=args.db)

    # 先把每个人名对应的所有“脸的图片块”收集起来
    id_to_crops = {}

    # 走遍所有图片路径
    for img_path in tqdm(list(iter_images(args.data_dir)), desc="Scanning images"):
        # 用图片所在的上一级文件夹名当作“这个人的名字”
        label = Path(img_path).parent.name

        # 读入图片（BGR 格式）
        img = cv2.imread(img_path)
        if img is None:
            # 读不到就跳过
            continue

        # 在图片里找人脸，并剪下来（得到很多小图，都是脸）
        crops = detector.crop_faces(img)
        if not crops:
            # 这张图片里没有脸，跳过
            continue

        # 把这张图片里的所有“脸小图”放到这个人的列表里
        if label not in id_to_crops:
            id_to_crops[label] = []
        id_to_crops[label].extend(crops)

    # 现在开始真正“录入”到数据库：
    for name, crops in id_to_crops.items():
        # 把所有脸小图变成“数字指纹”（向量）
        embs = embedder.get_embeddings(crops)
        if embs.size:
            # 存到数据库里：这个人对应很多条向量
            db.add(name, embs)
            print(f"Enrolled {name}: {embs.shape[0]} embeddings")


if __name__ == "__main__":
    main()
