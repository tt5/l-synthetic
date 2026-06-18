#!/usr/bin/env python3
"""generate_dataset.py

Generate synthetic pictogram dataset with 7000 images per class.
Saves images as .npy files in class-specific directories.

Usage:
    uv run generate_dataset.py
"""

import os
import numpy as np
from publisher import World
from classifier import classify

OUTPUT_DIR = "data"
IMAGES_PER_CLASS = 7000
NUM_CLASSES = 11  # classes 0-9 + class 11 (unclassified as -1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for c in list(range(10)) + [11]:
        os.makedirs(os.path.join(OUTPUT_DIR, f"class_{c}"), exist_ok=True)

    counts = {c: 0 for c in list(range(10)) + [11]}
    total = 0
    world = World()

    print(f"Generating {IMAGES_PER_CLASS} images per class ({NUM_CLASSES} classes)...")

    while min(counts[c] for c in range(10)) < IMAGES_PER_CLASS:
        grid, metadata = world.get_random_image()
        label = classify(metadata, grid)
        if label == -1:
            label = 11

        if label < 11 and counts[label] >= IMAGES_PER_CLASS:
            continue

        path = os.path.join(OUTPUT_DIR, f"class_{label}", f"{counts[label]:06d}.npy")
        np.save(path, grid)
        counts[label] += 1
        total += 1

        if total % 100 == 0:
            print(f"  Generated {total} images, counts: {counts}")

    print(f"Done. Total: {total} images.")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
