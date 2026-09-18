# GeoLite3D: Lightweight Monocular 3D Object Detection for Autonomous Driving

GeoLite3D is a lightweight, geometry-guided monocular 3D object detection framework designed for autonomous driving applications. Powered by a **YOLOv10** backbone and custom multi-task prediction heads, GeoLite3D estimates 3D bounding boxes ($X, Y, Z, h, w, l, r_y$) directly from a single RGB image and camera calibration metrics.

---

## Key Features & Research Highlights

- **YOLOv10 NMS-Free Backbone**: Utilizes YOLOv10 feature extractor to eliminate NMS latency while maintaining high FPS for real-time 3D object detection.
- **Geometry-Aware Multi-Task Heads**: Predicts Heatmaps, Depth ($Z$), 3D Dimensions ($h, w, l$), and Orientation angles ($\sin, \cos$).
- **KITTI Benchmark Data Pipeline**: Built-in automated dataset downloader, exploratory data analysis (EDA) suite, and 70/15/15 train/val/test data splitting logic.
- **Multi-Task Loss Formulation**: Employs Focal Loss for keypoint detection alongside L1 and Smooth L1 Loss for depth, spatial dimension, and orientation regression.
- **Camera Back-Projection Decoder**: 3D bounding box decoder leveraging the camera calibration matrix ($P_2$) for real-world 3D localization.

---

## Repository Structure

```text
GeoLite3D/
├── data/                  # KITTI dataset directory and split files
├── logs/                  # Development logs and training progress tracking
├── results/               # Inference predictions and KITTI-formatted outputs
├── src/
│   ├── dataset/           # KITTI dataset loader, custom collate, and split generator
│   ├── eda/               # Exploratory data analysis scripts
│   ├── evaluation/        # KITTI official format 3D AP evaluation metrics
│   ├── inference/         # Top-K peak detection and 2D-to-3D back-projection decoder
│   ├── models/            # YOLOv10 feature extractor & multi-task 3D heads
│   ├── training/          # Multi-task loss formulations & PyTorch training engine
│   └── utils/             # Automated KITTI dataset downloader script
├── .gitignore             # Git exclusion rules (data, weights, virtualenv)
└── README.md              # Project documentation
```

---

## Installation & Environment Setup

### 1. Clone the Repository:

```bash

git clone https://github.com/wolverSTW/GeoLite3D.git
cd GeoLite3D

```

### 2. Set Up Virtual Environment:

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / Git Bash
source .venv/Scripts/activate
```

### 3. Install Dependencies:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install ultralytics pillow matplotlib
```

---

## Dataset Pipeline (KITTI 3D)

### 1. Automated Dataset Download

Download KITTI benchmark components directly using the utility downloader:

- Mini Mode (Labels + Calibration only):

```bash
python src/utils/download_kitti.py --mode mini
```

- Full Mode (Images + Labels + Calibration):

```bash
python src/utils/download_kitti.py --mode full
```

### 2. Exploratory Data Analysis (EDA)

Analyze class distribution and spatial statistics across the dataset:

```bash
PYTHONPATH=. python src/eda/analyze_kitti.py
```

### 3. Generate Splits

Create the 70/15/15 train/val/test splits:

```bash
PYTHONPATH=. python src/dataset/kitti_dataset.py
```

---

## Execution Guide

### 1. Model Forward Pass Verification

Verify YOLOv10 backbone integration and feature map output shapes:

```bash
PYTHONPATH=. python src/models/geolite3d_model.py
```

### 2. Training Pipeline

Run the multi-task training engine:

```bash
PYTHONPATH=. python src/training/trainer.py
```

### 3. Inference & 3D Box Decoding

Execute 3D bounding box post-processing and back-projection decoder:

```bash
PYTHONPATH=. python src/inference/predict.py
```

### 4. KITTI Format Evaluation

Format model output boxes into official KITTI evaluation label files:

```bash
PYTHONPATH=. python src/evaluation/eval_kitti.py
```

---

## License & Citation

Developed as part of MSc Dissertation research in Autonomous Driving Vision Systems.
