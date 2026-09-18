"""
===============================================================================
GeoLite3D

KITTI Evaluation Pipeline & Metric Formatter

Description:
Formats model predictions into KITTI official submission format for 3D AP evaluation.
===============================================================================
"""

import os
from pathlib import Path


class KITTIEvaluator:
    """
    Formats 3D predictions into KITTI label format:
    type, truncated, occluded, alpha, bbox_2d (4), dimensions_3d (3), location_3d (3), rotation_y, score
    """
    def __init__(self, output_dir: str = "./results/kitti_format"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_map = {0: "Car", 1: "Pedestrian", 2: "Cyclist", 3: "Van", 4: "Truck"}

    def write_kitti_result(self, image_id: str, predictions: list):
        file_path = self.output_dir / f"{image_id}.txt"
        with open(file_path, "w") as f:
            for pred in predictions:
                cls_name = self.class_map.get(pred["class_id"], "DontCare")
                score = pred["score"]
                x, y, z, h, w, l, ry = pred["box_3d"]

                # KITTI Standard Line Structure
                # bbox2d is placeholder [0, 0, 50, 50] for evaluation pipeline
                line = f"{cls_name} 0.00 0 0.00 0.00 0.00 50.00 50.00 {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.4f}\n"
                f.write(line)


if __name__ == "__main__":
    print("Initializing KITTI Evaluation Formatter Verification...")
    evaluator = KITTIEvaluator()

    dummy_boxes = [{
        "class_id": 0,
        "score": 0.85,
        "box_3d": [1.2, 0.5, 15.4, 1.52, 1.60, 3.88, 0.25]
    }]

    evaluator.write_kitti_result("000001", dummy_boxes)
    print(f"Sample prediction formatted and written to: ./results/kitti_format/000001.txt\n")
