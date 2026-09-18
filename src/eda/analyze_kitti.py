"""
===============================================================================
GeoLite3D

KITTI Exploratory Data Analysis (EDA) Script

Description:
Analyze class distributions, distance (Z) distributions, objects per image,
and class-wise distance profiles dynamically across ALL KITTI label classes.
===============================================================================
"""

from pathlib import Path
import math
from collections import Counter


class KITTIExploratoryAnalysis:
    def __init__(self, label_dir: str = "./data/kitti/training/label_2"):
        self.label_dir = Path(label_dir)
        self.class_counts = Counter()
        self.distances_by_class = {}
        self.objects_per_image = []

    def parse_labels(self):
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found at: {self.label_dir}")

        label_files = list(self.label_dir.glob("*.txt"))
        print(f"Parsing {len(label_files)} label files for EDA...")

        for file_path in label_files:
            with open(file_path, "r") as f:
                lines = f.readlines()

            num_objects = len(lines)
            self.objects_per_image.append(num_objects)

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                obj_type = parts[0]
                # Extract 3D coordinates (X, Y, Z) from KITTI label format
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                
                # Euclidean 3D Distance
                distance = math.sqrt(x**2 + y**2 + z**2)

                self.class_counts[obj_type] += 1

                if obj_type not in self.distances_by_class:
                    self.distances_by_class[obj_type] = []
                self.distances_by_class[obj_type].append(distance)

    def print_summary(self):
        print("\n==================================================")
        print(" GeoLite3D - KITTI Exploratory Data Analysis (EDA)")
        print("==================================================")

        # Step 8: Class Distribution (All Classes)
        print("\n[Step 8] Object Class Distribution (All Classes):")
        print("--------------------------------------------------")
        for obj_cls, count in self.class_counts.most_common():
            print(f"  - {obj_cls:<15}: {count:>6} objects")

        # Step 10: Objects per Image Statistics
        if self.objects_per_image:
            avg_objs = sum(self.objects_per_image) / len(self.objects_per_image)
            max_objs = max(self.objects_per_image)
            min_objs = min(self.objects_per_image)
            print("\n[Step 10] Objects Per Image Analysis:")
            print("--------------------------------------------------")
            print(f"  - Total Images      : {len(self.objects_per_image)}")
            print(f"  - Average Objects   : {avg_objs:.2f} per image")
            print(f"  - Min / Max Objects : {min_objs} / {max_objs}")

        # Step 9 & 11: Class-wise Distance Analysis (Dynamic for ALL Classes)
        print("\n[Step 9 & 11] Class-wise Distance (Z-Depth) Profile (All Classes):")
        print("--------------------------------------------------")
        # Sort classes by object count (most common first)
        for obj_cls, _ in self.class_counts.most_common():
            dists = self.distances_by_class[obj_cls]
            avg_dist = sum(dists) / len(dists)
            min_dist = min(dists)
            max_dist = max(dists)
            print(f"  - {obj_cls:<15} -> Count: {len(dists):>6} | Avg: {avg_dist:>5.2f}m | Min: {min_dist:>5.2f}m | Max: {max_dist:>5.2f}m")

        print("==================================================\n")


def main():
    eda = KITTIExploratoryAnalysis()
    eda.parse_labels()
    eda.print_summary()


if __name__ == "__main__":
    main()
