import os

def verify_kitti_structure(base_path="data/kitti/training"):
    subfolders = ["image_2", "label_2", "calib"]
    counts = {}

    print(f"=== Checking KITTI Dataset Structure at: {base_path} ===")
    
    if not os.path.exists(base_path):
        print(f"[ERROR] Path does not exist: {base_path}")
        return False

    all_valid = True
    for folder in subfolders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            counts[folder] = len(files)
            print(f"[OK] Folder '{folder}' found with {len(files)} files.")
        else:
            print(f"[ERROR] Missing folder: {folder_path}")
            all_valid = False

    if all_valid:
        num_images = counts["image_2"]
        num_labels = counts["label_2"]
        num_calibs = counts["calib"]

        if num_images == num_labels == num_calibs and num_images > 0:
            print(f"\n[SUCCESS] Dataset structure is valid! Total samples: {num_images}")
        else:
            print(f"\n[WARNING] File count mismatch! Images: {num_images}, Labels: {num_labels}, Calibs: {num_calibs}")

    return all_valid

if __name__ == "__main__":
    verify_kitti_structure()
