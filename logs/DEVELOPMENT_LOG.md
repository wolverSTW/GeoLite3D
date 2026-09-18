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
