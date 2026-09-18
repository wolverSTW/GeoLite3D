"""
===============================================================================
GeoLite3D

KITTI Dataset Downloader

Description:
Download and verify KITTI dataset files.

Modes
-----
mini:
    Download labels and calibration only.

full:
    Download images, labels and calibration.

Usage
-----
python src/utils/download_kitti.py --mode mini
python src/utils/download_kitti.py --mode full
===============================================================================
"""

from pathlib import Path
import argparse
import urllib.request
import zipfile

from tqdm import tqdm


KITTI_URLS = {
    "images":
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",

    "labels":
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",

    "calib":
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
}


class DownloadProgressBar(tqdm):
    """
    Progress bar helper for urlretrieve.
    """

    def update_to(
        self,
        block_num=1,
        block_size=1,
        total_size=None
    ):
        if total_size is not None:
            self.total = total_size

        self.update(block_num * block_size - self.n)


class KITTIDownloadManager:
    """
    Download and organize KITTI dataset.
    """

    def __init__(self, output_dir: str):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ------------------------------------------------------------------

    def download_file(
        self,
        url: str,
        save_path: Path
    ):
        """
        Download file with progress bar.
        """

        print(f"\nDownloading -> {save_path.name}")

        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=save_path.name
        ) as progress:

            urllib.request.urlretrieve(
                url,
                filename=save_path,
                reporthook=progress.update_to
            )

    # ------------------------------------------------------------------

    def extract_zip(
        self,
        zip_path: Path
    ):
        """
        Extract zip archive.
        """

        print(f"Extracting -> {zip_path.name}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.output_dir)

        zip_path.unlink()

        print(f"Done -> {zip_path.name}")

    # ------------------------------------------------------------------

    def download_component(
        self,
        component_name: str
    ):
        """
        Download one KITTI component.
        """

        url = KITTI_URLS[component_name]

        zip_path = self.output_dir / f"{component_name}.zip"

        self.download_file(
            url,
            zip_path
        )

        self.extract_zip(
            zip_path
        )

    # ------------------------------------------------------------------

    def verify_structure(self):
        """
        Verify KITTI folders.
        """

        print("\nChecking dataset structure...")

        image_dir = self.output_dir / "training" / "image_2"
        label_dir = self.output_dir / "training" / "label_2"
        calib_dir = self.output_dir / "training" / "calib"

        if not image_dir.exists():
            image_dir = self.output_dir / "image_2"

        if not label_dir.exists():
            label_dir = self.output_dir / "label_2"

        if not calib_dir.exists():
            calib_dir = self.output_dir / "calib"

        image_count = (
            len(list(image_dir.glob("*")))
            if image_dir.exists()
            else 0
        )

        label_count = (
            len(list(label_dir.glob("*")))
            if label_dir.exists()
            else 0
        )

        calib_count = (
            len(list(calib_dir.glob("*")))
            if calib_dir.exists()
            else 0
        )

        print("--------------------------------")
        print("KITTI Dataset Verification")
        print("--------------------------------")
        print(f"Images : {image_count}")
        print(f"Labels : {label_count}")
        print(f"Calibs : {calib_count}")
        print("--------------------------------")

    # ------------------------------------------------------------------

    def run(
        self,
        mode: str
    ):
        """
        Execute download.
        """

        print("\n======================================")
        print("GeoLite3D KITTI Downloader")
        print("======================================")

        if mode == "mini":

            print("\nMode: MINI")
            print("Downloading Labels + Calibration")

            self.download_component("labels")
            self.download_component("calib")

        elif mode == "full":

            print("\nMode: FULL")
            print("Downloading Images + Labels + Calibration")

            self.download_component("images")
            self.download_component("labels")
            self.download_component("calib")

        else:

            raise ValueError(
                "Mode must be mini or full"
            )

        self.verify_structure()

        print("\nKITTI setup completed ✅")


def main():

    parser = argparse.ArgumentParser(
        description="GeoLite3D KITTI Downloader"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="mini",
        choices=["mini", "full"]
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/kitti"
    )

    args = parser.parse_args()

    downloader = KITTIDownloadManager(
        output_dir=args.output_dir
    )

    downloader.run(
        mode=args.mode
    )


if __name__ == "__main__":
    main()
