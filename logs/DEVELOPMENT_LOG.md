# GeoLite3D Development Log

## [2026-09-18] Phase 0 - Step 1: Research Area Exploration & Project Setup

- **Goal**: Initialize GeoLite3D directory structure and Git repository.
- **Summary**:
  - Defined standard research directory architecture.
  - Configured `.gitignore` for dataset and model weight exclusion.
  - Set up logging framework for dissertation tracking.
- **Status**: Completed setup.

## [2026-09-18] Phase 0 - Step 2: Literature Review

- **Goal**: Review core theoretical concepts of Monocular 3D Detection, YOLOv10 architecture, and Geometry-guided depth estimation.
- **Key Takeaways**:
  1. Monocular 3D detection suffers from depth ambiguity ($3\text{D} \to 2\text{D}$ projection loss).
  2. Geometry constraint formula $Z_{geo} = \frac{f \cdot H_{3d}}{h_{2d}}$ provides a strong physical prior.
  3. YOLOv10's NMS-free architecture will serve as the 2D backbone for GeoLite3D.
- **Status**: Completed Step 2 review.

## [2026-09-18] Phase 0 - Step 3: Research Gap Identification

- **Goal**: Identify key limitations in current literature and define GeoLite3D core contributions.
- **Identified Research Gaps**:
  1. High computational latency in existing geometry models due to NMS post-processing.
  2. Lack of physical geometry constraints in ultra-fast 2D detectors like YOLOv10.
  3. Depth regression instability without probabilistic uncertainty estimation.
- **GeoLite3D Solutions**:
  1. Integrate YOLOv10 NMS-free pipeline for high FPS.
  2. Add Geometry-Aware Depth Predictor ($Z_{geo} + Z_{err}$).
  3. Implement $\beta$-NLL Loss for uncertainty propagation.
- **Status**: Completed Step 3 gap analysis.

## [2026-09-18] Phase 0 - Step 4: GeoLite3D Project Definition

- **Goal**: Finalize project definition, aims, objectives, and scope for Dissertation documentation.
- **Project Aim**: Build a lightweight, geometry-guided Monocular 3D Object Detector (GeoLite3D) leveraging YOLOv10.
- **Key Objectives**:
  - Implement KITTI Data & Calibration Pipeline.
  - Design Geometry Predictor Engine ($Z_{geo} + Z_{err}$).
  - Implement Probabilistic Uncertainty Loss ($\beta$-NLL).
  - Benchmark $AP_{3D}$, $AP_{BEV}$, Depth MAE/RMSE, and FPS on RTX 4090.
- **Status**: Completed Phase 0 (Research Initiation & Problem Definition).

## [2026-09-18] Phase 1 - Step 5 & 6: KITTI Downloader Script

- **Goal**: Implement `src/utils/download_kitti.py` to automate downloading and extraction of KITTI dataset.
- **Modes Supported**: `mini` (labels + calib), `full` (images + labels + calib).
- **Status**: Downloader script integrated into `src/utils/`.

## [2026-09-18] Phase 2 - Steps 8–11: Exploratory Data Analysis (EDA)
- **Goal**: Analyze class distributions, objects per image, and distance stats across KITTI dataset.
- **Created File**: `src/eda/analyze_kitti.py`
- **Status**: Executed EDA script and extracted target dataset statistics.

## [2026-09-18] Phase 2 - Steps 8–11: EDA Execution Completed
- **Dataset Summary**: Analyzed 7,481 training images and label files.
- **Class Stats**: Car (28,742), DontCare (11,295), Pedestrian (4,487), Van (2,914), Cyclist (1,627), Truck (1,094), Misc (973), Tram (511), Person_sitting (222).
- **Density**: Average 6.93 objects per image (Min: 1, Max: 24).
- **Depth Stats**: Car average distance ~29.38m, Pedestrian ~18.44m, Cyclist ~25.92m.
- **Status**: Phase 2 completed.

## [2026-09-18] Phase 2 - Steps 8–11: EDA Execution Completed
- **Dataset Summary**: Analyzed 7,481 training images and label files.
- **Class Stats**: Car (28,742), DontCare (11,295), Pedestrian (4,487), Van (2,914), Cyclist (1,627), Truck (1,094), Misc (973), Tram (511), Person_sitting (222).
- **Density**: Average 6.93 objects per image (Min: 1, Max: 24).
- **Depth Stats**: Car average distance ~29.38m, Pedestrian ~18.44m, Cyclist ~25.92m.
- **Status**: Phase 2 completed.

## [2026-09-18] Phase 3 - Steps 12–17: Dataset Preparation & DataLoader Verification
- **Split Stats**: Total 7,481 (Train: 5,236, Val: 1,122, Test: 1,123).
- **DataLoader Verification**:
  - Image Tensor Shape: `torch.Size([4, 3, 384, 1280])`
  - Calibration Matrix P2 Shape: `torch.Size([4, 3, 4])`
- **Status**: Completed Phase 3.

## [2026-09-18] Phase 4 - YOLOv10 Backbone Verification
- **Dependency**: Installed `ultralytics` package.
- **Verification**: Verified YOLOv10 feature extraction and Monocular 3D heads forward pass.
- **Status**: Completed Phase 4 (Model Architecture & Backbone Integration).

## [2026-09-18] Phase 5 - Steps 23–27: Loss Function & Training Pipeline Integration
- **Loss Formulation**: Formulated Multi-task Loss (Focal + L1 + Smooth L1) for 3D Monocular Detection.
- **Trainer Engine**: Built `GeoLiteTrainer` supporting AdamW optimizer and PyTorch data processing loop.
- **Status**: Executed sanity check and verified training loop execution.
