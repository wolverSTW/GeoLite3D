"""
===============================================================================
GeoLite3D

KITTI PyTorch Dataset and Data Splitting Module

Description:
Custom PyTorch Dataset class for KITTI 3D Object Detection.
Supports 70/15/15 Train/Val/Test splitting, image preprocessing (resize/normalize),
and label encoding.
===============================================================================
"""

from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from src.utils.kitti_utils import CalibrationReader, LabelReader


# Mapping classes to numerical IDs
CLASS_TO_ID = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Van": 3,
    "Truck": 4,
    "Person_sitting": 5,
    "Tram": 6,
    "Misc": 7
}


def create_kitti_splits(
    data_dir: str = "./data/kitti/training",
    split_dir: str = "./data/kitti/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Splits all KITTI sample IDs into Train (70%), Val (15%), and Test (15%).
    Saves split file lists (.txt) under data/kitti/splits/
    """
    data_path = Path(data_dir)
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list((data_path / "image_2").glob("*.png")))
    sample_ids = [f.stem for f in image_files]

    random.seed(seed)
    random.shuffle(sample_ids)

    total = len(sample_ids)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:n_train + n_val]
    test_ids = sample_ids[n_train + n_val:]

    # Write split IDs to text files
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(split_path / f"{split_name}.txt", "w") as f:
            for sample_id in ids:
                f.write(f"{sample_id}\n")

    print("==================================================")
    print(" GeoLite3D - Dataset Splitting Completed (70/15/15)")
    print("==================================================")
    print(f" Total Samples : {total}")
    print(f" Train Split   : {len(train_ids)} ({len(train_ids)/total*100:.1f}%)")
    print(f" Val Split     : {len(val_ids)} ({len(val_ids)/total*100:.1f}%)")
    print(f" Test Split    : {len(test_ids)} ({len(test_ids)/total*100:.1f}%)")
    print("==================================================\n")


class GeoLiteKITTIDataset(Dataset):
    """
    PyTorch Dataset for GeoLite3D KITTI Object Detection
    """
    def __init__(
        self,
        data_dir: str = "./data/kitti/training",
        split_file: str = "./data/kitti/splits/train.txt",
        img_size: tuple = (384, 1280)
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        # Read sample IDs for the split
        with open(split_file, "r") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

        # Image transformations: Resize & ImageNet Normalization
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sample_id = self.sample_ids[index]

        img_path = self.data_dir / "image_2" / f"{sample_id}.png"
        label_path = self.data_dir / "label_2" / f"{sample_id}.txt"
        calib_path = self.data_dir / "calib" / f"{sample_id}.txt"

        # Load Image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # Parse Calibration Matrix
        calib = CalibrationReader(calib_path)
        p2_matrix = torch.tensor(calib.P2, dtype=torch.float32)

        # Parse Labels
        labels = LabelReader(label_path).read_labels(filter_dontcare=True)

        gt_boxes_3d = []
        gt_classes = []

        for obj in labels:
            cls_name = obj["type"]
            cls_id = CLASS_TO_ID.get(cls_name, -1)
            if cls_id == -1:
                continue

            # 3D Box Parameters: [x, y, z, h, w, l, ry]
            loc = obj["location"]
            dim = obj["dimensions"]
            ry = obj["rotation_y"]
            box_3d = [loc[0], loc[1], loc[2], dim[0], dim[1], dim[2], ry]

            gt_boxes_3d.append(box_3d)
            gt_classes.append(cls_id)

        return {
            "sample_id": sample_id,
            "image": img_tensor,
            "P2": p2_matrix,
            "gt_boxes_3d": torch.tensor(gt_boxes_3d, dtype=torch.float32) if gt_boxes_3d else torch.empty((0, 7)),
            "gt_classes": torch.tensor(gt_classes, dtype=torch.long) if gt_classes else torch.empty((0,), dtype=torch.long)
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable number of 3D objects per image
    """
    sample_ids = [item["sample_id"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    p2_matrices = torch.stack([item["P2"] for item in batch])
    gt_boxes_3d = [item["gt_boxes_3d"] for item in batch]
    gt_classes = [item["gt_classes"] for item in batch]

    return {
        "sample_ids": sample_ids,
        "images": images,
        "P2": p2_matrices,
        "gt_boxes_3d": gt_boxes_3d,
        "gt_classes": gt_classes
    }


if __name__ == "__main__":
    # Create 70/15/15 splits
    create_kitti_splits()

    # Test DataLoader Integration
    dataset = GeoLiteKITTIDataset(split_file="./data/kitti/splits/train.txt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    print("Verifying DataLoader Batch Extraction...")
    for batch in dataloader:
        print(f" Batch Images Tensor Shape: {batch['images'].shape}")
        print(f" Batch P2 Matrix Shape   : {batch['P2'].shape}")
        print(f" Sample IDs in Batch     : {batch['sample_ids']}")
        print(f" Objects in 1st Sample   : {batch['gt_boxes_3d'][0].shape[0]} boxes")
        break
