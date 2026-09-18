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

## [2026-09-18] Phase 1 - Steps 5, 6, 7: Dataset Selection, Acquisition & Verification
- **Goal**: Verify KITTI 3D dataset structure (`image_2`, `label_2`, `calib`) using an automated verification script.
- **Created File**: `src/utils/check_kitti_dataset.py`
- **Status**: Completed Phase 1 dataset setup.
