#!/usr/bin/env python3
"""generate_dataset.py

Generate synthetic pictogram dataset.
Saves images as .npy files in class-specific directories.

Usage:
    uv run generate_dataset.py [--count N] [--data-dir data]
"""

import argparse
import os
import numpy as np
from publisher import World
from classifier import classify

DEFAULT_DATA_DIR = "data"
DEFAULT_IMAGES_PER_CLASS = 7000
NUM_CLASSES = 11  # classes 0-9 + class 11 (unclassified as -1)


def count_existing(data_dir):
    """Count existing images per class directory."""
    counts = {}
    for c in list(range(10)) + [11]:
        class_dir = os.path.join(data_dir, f"class_{c}")
        if os.path.exists(class_dir):
            counts[c] = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
        else:
            counts[c] = 0
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=DEFAULT_IMAGES_PER_CLASS,
                        help="Target images per class (default: 7000)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Output directory (default: data)")
    args = parser.parse_args()

    data_dir = args.data_dir
    images_per_class = args.count

    os.makedirs(data_dir, exist_ok=True)
    for c in list(range(10)) + [11]:
        os.makedirs(os.path.join(data_dir, f"class_{c}"), exist_ok=True)

    counts = count_existing(data_dir)
    total_existing = sum(counts.values())
    print(f"Existing images: {total_existing}")
    print(f"Per-class: {counts}")
    print(f"Target: {images_per_class} per class ({NUM_CLASSES} classes)")

    total = 0
    world = World()

    while min(counts[c] for c in range(10)) < images_per_class:
        grid, metadata = world.get_random_image()
        label = classify(metadata, grid)
        if label == -1:
            label = 11

        if label <= 11 and counts[label] >= images_per_class:
            continue

        path = os.path.join(data_dir, f"class_{label}", f"{counts[label]:06d}.npy")
        np.save(path, grid)
        counts[label] += 1
        total += 1

        if total % 100 == 0:
            print(f"  Generated {total} new images, counts: {counts}")

    print(f"Done. Generated {total} new images.")
    print(f"Total per-class: {counts}")


if __name__ == "__main__":
    main()
