import argparse
import os
import numpy as np
from scipy import ndimage
from src.utils import postprocess

DEFAULT_DATA_DIR = "data"
DEFAULT_COUNT = 5000


def count_existing(data_dir):
    """Count existing images per class directory."""
    class_dir = os.path.join(data_dir, "class_3")
    if os.path.exists(class_dir):
        return len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
    return 0


def generate_to_dir(data_dir, count):
    """Generate images into data_dir/class_3 until count new images are created."""
    class_dir = os.path.join(data_dir, "class_3")
    os.makedirs(class_dir, exist_ok=True)

    existing = count_existing(data_dir)
    print(f"Generating {count} images in {class_dir} (existing files: {existing})...")

    total = 0
    for i in range(count):
        imgr = np.random.randint(0, 2, (26, 26))
        imgra = ndimage.binary_dilation(imgr)
        imgra = ndimage.binary_erosion(imgra)
        imgra = imgra * (-1) + 1

        imgrb = ndimage.binary_erosion(imgr)
        imgrb = ndimage.binary_dilation(imgrb)
        imgrb = ndimage.binary_dilation(imgrb)

        imgr = imgra + imgrb
        imgr = np.clip(imgr, 0, 1)
        imgr = ndimage.binary_erosion(imgr)
        imgr = ndimage.binary_dilation(imgr)
        imgr, _ = postprocess(imgr)

        filename = f"{existing + i:06d}.npy"
        np.save(os.path.join(class_dir, filename), imgr.astype(np.uint8))

        total += 1
        if total % 1000 == 0:
            print(f"  Saved {total} / {count}")

    print(f"Done. Total files in {class_dir}: {existing + count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of images to generate")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Training output directory")
    args = parser.parse_args()

    generate_to_dir(args.data_dir, args.count)


if __name__ == "__main__":
    main()
