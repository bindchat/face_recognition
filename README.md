# YOLO 人脸识别系统

使用 YOLO 进行人脸检测，结合 InsightFace 提取特征，实现多人（≥5）的人脸识别。支持图片/视频/摄像头识别，并提供人脸库入库工具。

## 目录结构
```
src/
  detector/        # YOLO 人脸检测
  embedding/       # InsightFace 人脸特征
  utils/           # 可视化/权重下载/人脸库
  cli/             # 命令行：入库与识别
data/
models/
scripts/
```

## 环境安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_weights.py  # 下载 YOLO 人脸检测权重
```

若使用 GPU，请确保已安装匹配 CUDA 的 PyTorch（可从 `https://pytorch.org` 获取命令）。

## 数据准备（至少5人）
按身份建子目录，每个目录放若干该人的人脸图片：
```
data/enroll/
  Alice/*.jpg
  Bob/*.jpg
  Charlie/*.jpg
  Diana/*.jpg
  Evan/*.jpg
  ...
```

## 入库（构建人脸数据库）
```bash
python -m src.cli.enroll data/enroll --db data/faces_db.json --weights yolov8n-face.pt
```

## 识别
- 图片：
```bash
python -m src.cli.recognize --image path/to/img.jpg --db data/faces_db.json --weights yolov8n-face.pt --output runs/out.jpg
```

- 视频文件：
```bash
python -m src.cli.recognize --video path/to/video.mp4 --db data/faces_db.json --weights yolov8n-face.pt --output runs/out.mp4
```

- 摄像头：
```bash
python -m src.cli.recognize --webcam --db data/faces_db.json --weights yolov8n-face.pt
```

## 说明
- YOLO 负责检测人脸框；InsightFace 提取512维特征；人脸库使用类中心（质心）+ 余弦相似度进行匹配。
- 阈值可在 `src/utils/db.py` 的 `search` 中调节（默认较稳健）。
- 权重默认下载到项目根目录 `yolov8n-face.pt`，可通过 `--weights` 指定。
- 如在无显示环境下运行，请为图片/视频使用 `--output` 保存结果，避免 `cv2.imshow`。
