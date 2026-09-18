"""
===============================================================================
GeoLite3D

Inference Engine & 3D Box Decoder Module

Description:
Decodes GeoLite3D prediction heads (heatmap, depth, dimensions, orientation)
into 3D bounding boxes and projects them onto image coordinates.
===============================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np


class GeoLiteDecoder:
    """
    Decodes multi-task prediction maps into 3D Bounding Boxes
    """
    def __init__(self, topk: int = 20, confidence_threshold: float = 0.3):
        self.topk = topk
        self.conf_thresh = confidence_threshold

    def decode(self, predictions: dict, p2_matrix: torch.Tensor) -> list:
        """
        Decode output tensors for a single sample.
        """
        heatmap = torch.sigmoid(predictions["heatmap"])  # [1, Num_Classes, H, W]
        depth = predictions["depth"]                    # [1, 1, H, W]
        dim = predictions["dim"]                        # [1, 3, H, W]
        ori = predictions["orientation"]                # [1, 2, H, W]

        # Find max confidence score across classes
        scores, class_ids = torch.max(heatmap, dim=1)  # [1, H, W]

        # Extract Top-K predictions
        b, h, w = scores.shape
        scores_flat = scores.view(b, -1)
        topk_scores, topk_inds = torch.topk(scores_flat, self.topk, dim=1)

        decoded_boxes = []

        for i in range(self.topk):
            score = topk_scores[0, i].item()
            if score < self.conf_thresh:
                continue

            idx = topk_inds[0, i].item()
            cy = idx // w
            cx = idx % w

            cls_id = class_ids[0, cy, cx].item()
            z_depth = depth[0, 0, cy, cx].item()
            box_dim = dim[0, :, cy, cx].tolist()  # [h, w, l]
            ori_vec = ori[0, :, cy, cx]
            heading = torch.atan2(ori_vec[0], ori_vec[1]).item()

            # Back-project 2D center to 3D Camera Coordinate System [X, Y, Z]
            # P2: [[fx, 0, cx, tx], [0, fy, cy, ty], [0, 0, 1, tz]]
            fx = p2_matrix[0, 0].item()
            fy = p2_matrix[1, 1].item()
            cx_p2 = p2_matrix[0, 2].item()
            cy_p2 = p2_matrix[1, 2].item()

            x_3d = (cx - cx_p2) * z_depth / fx
            y_3d = (cy - cy_p2) * z_depth / fy

            decoded_boxes.append({
                "class_id": cls_id,
                "score": score,
                "box_3d": [x_3d, y_3d, z_depth, box_dim[0], box_dim[1], box_dim[2], heading]
            })

        return decoded_boxes


if __name__ == "__main__":
    from src.models.geolite3d_model import GeoLite3D

    print("Initializing Inference Engine Check...")
    model = GeoLite3D(num_classes=8, yolo_variant="yolov10n.pt")
    model.eval()

    decoder = GeoLiteDecoder(topk=10, confidence_threshold=0.1)

    # Dummy Input and Calibration Matrix Setup
    dummy_img = torch.randn(1, 3, 384, 1280)
    dummy_p2 = torch.tensor([
        [721.5377, 0.0, 609.5593, 44.8572],
        [0.0, 721.5377, 172.8540, 0.2163],
        [0.0, 0.0, 1.0, 0.0027]
    ], dtype=torch.float32)

    with torch.no_grad():
        preds = model(dummy_img)
        boxes = decoder.decode(preds, dummy_p2)

    print("\n==================================================")
    print(" GeoLite3D - Inference 3D Decoder Output")
    print("==================================================")
    print(f" Detected 3D Objects Count: {len(boxes)}")
    if boxes:
        print(f" Sample Detected Box [0]  : {boxes[0]}")
    print("==================================================\n")
