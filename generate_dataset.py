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
from src.utils import postprocess
from publisher import World
from classifier import classify

DEFAULT_DATA_DIR = "data"
DEFAULT_TEST_DATA_DIR = "test_data"
DEFAULT_IMAGES_PER_CLASS = 7000
NUM_CLASSES = 3 # unclassified samples are skipped, not saved as a class


def count_existing(data_dir):
    """Count existing images per class directory."""
    counts = {}
    for c in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, f"class_{c}")
        if os.path.exists(class_dir):
            counts[c] = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
        else:
            counts[c] = 0
    return counts


def generate_to_dir(data_dir, images_per_class, world):
    """Generate images into data_dir until each class has images_per_class images."""
    os.makedirs(data_dir, exist_ok=True)
    for c in range(NUM_CLASSES):
        os.makedirs(os.path.join(data_dir, f"class_{c}"), exist_ok=True)

    counts = count_existing(data_dir)
    print(f"Generating in {data_dir}: target {images_per_class} per class")

    total = 0
    skipped = 0
    while min(counts[c] for c in range(NUM_CLASSES)) < images_per_class:
        grid, metadata = world.get_random_image()
        grid, metadata["centered"] = postprocess(grid)
        label = classify(metadata, grid)

        if label == -1:
            skipped += 1
            if skipped % 100 == 0:
                print(f"  Generated {total} new images, counts: {counts}")
            continue

        if counts[label] >= images_per_class:
            continue

        path = os.path.join(data_dir, f"class_{label}", f"{counts[label]:06d}.npy")
        np.save(path, grid)
        counts[label] += 1
        total += 1

        if total % 100 == 0:
            print(f"  Generated {total} new images, counts: {counts}")

    print(f"Done. Generated {total} new images in {data_dir}.")
    print(f"Skipped {skipped} unclassified images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=DEFAULT_IMAGES_PER_CLASS,
                        help="Target images per class (default: 7000)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Training output directory (default: data)")
    parser.add_argument("--test-data", action="store_true",
                        help="Also generate separate test data in test_data/")
    args = parser.parse_args()

    world = World()

    # Generate training data
    generate_to_dir(args.data_dir, args.count, world)

    # Generate test data if requested
    if args.test_data:
        test_dir = os.path.join(os.path.dirname(args.data_dir), DEFAULT_TEST_DATA_DIR)
        print(f"\n=== Generating test data in {test_dir} ===")
        generate_to_dir(test_dir, args.count, world)


if __name__ == "__main__":
    main()
