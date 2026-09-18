"""
===============================================================================
GeoLite3D

Loss Function Formulation and PyTorch Training Pipeline

Description:
Multi-task Loss (Focal Loss + Smooth L1 Loss) and Training Loop for GeoLite3D.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path


class GeoLite3DLoss(nn.Module):
    """
    Multi-Task Loss Function for GeoLite3D Monocular 3D Object Detection:
    - Heatmap Loss: Focal Loss (Target Centroids)
    - Depth Loss: L1 Loss
    - Dimension Loss: Smooth L1 Loss
    - Orientation Loss: Smooth L1 Loss (sin, cos)
    """
    def __init__(self, w_heatmap: float = 1.0, w_depth: float = 1.0, w_dim: float = 1.0, w_ori: float = 1.0):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_depth = w_depth
        self.w_dim = w_dim
        self.w_ori = w_ori

        self.l1_loss = nn.L1Loss(reduction="mean")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    def forward(self, predictions: dict, targets: dict) -> dict:
        pred_hp = predictions["heatmap"]
        pred_depth = predictions["depth"]
        pred_dim = predictions["dim"]
        pred_ori = predictions["orientation"]

        # Dummy Multi-Task Loss Calculation for Pipeline Verification
        loss_hp = self.w_heatmap * F.mse_loss(pred_hp, torch.zeros_like(pred_hp))
        loss_depth = self.w_depth * self.l1_loss(pred_depth, torch.ones_like(pred_depth))
        loss_dim = self.w_dim * self.smooth_l1(pred_dim, torch.ones_like(pred_dim))
        loss_ori = self.w_ori * self.smooth_l1(pred_ori, torch.zeros_like(pred_ori))

        total_loss = loss_hp + loss_depth + loss_dim + loss_ori

        return {
            "total_loss": total_loss,
            "loss_hp": loss_hp,
            "loss_depth": loss_depth,
            "loss_dim": loss_dim,
            "loss_ori": loss_ori
        }


class GeoLiteTrainer:
    """
    Trainer Pipeline Engine for GeoLite3D
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = GeoLite3DLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            losses = self.criterion(predictions, targets=batch)

            total_loss = losses["total_loss"]
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()

            if step % 5 == 0:
                print(f"Epoch [{epoch}] | Step [{step}/{len(self.train_loader)}] | Total Loss: {total_loss.item():.4f}")

        return running_loss / len(self.train_loader)


if __name__ == "__main__":
    from src.dataset.kitti_dataset import GeoLiteKITTIDataset, custom_collate_fn
    from src.models.geolite3d_model import GeoLite3D

    print("Initializing Trainer Pipeline Check...")
    
    # Initialize Dataset and Model
    train_dataset = GeoLiteKITTIDataset(split_file="./data/kitti/splits/train.txt")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

    model = GeoLite3D(num_classes=8, yolo_variant="yolov10n.pt")
    trainer = GeoLiteTrainer(model=model, train_loader=train_loader, val_loader=None)

    print("Executing 1-Epoch Sanity Verification Run...")
    avg_loss = trainer.train_one_epoch(epoch=1)
    print(f"\nVerification Completed! Average Loss: {avg_loss:.4f}\n")
