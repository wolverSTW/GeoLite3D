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
