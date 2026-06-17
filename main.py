#!/usr/bin/env python3
"""generate_synthetic_1.py

Generate 6000 synthetic images of the digit "1" in MNIST format.
Outputs to data/synthetic-1-train.csv and data/synthetic-1-test.csv.

Usage:
    uv run generate_synthetic_1.py
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 28
NUM_TRAIN = 5000
NUM_TEST = 1000


def generate_one():
    """Generate a single synthetic '1' image as 28x28 numpy array (0-255)."""
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    # Randomize parameters
    center_x = IMG_SIZE // 2 + random.randint(-2, 2)
    thickness = random.randint(1, 3)
    top_y = random.randint(2, 6)
    bottom_y = random.randint(22, 26)
    angle = random.uniform(-8, 8)  # degrees
    has_top_serif = random.random() < 0.3
    has_bottom_serif = random.random() < 0.2

    # Draw main vertical stroke
    if thickness == 1:
        draw.line([(center_x, top_y), (center_x, bottom_y)], fill=255, width=1)
    else:
        for offset in range(-(thickness // 2), thickness // 2 + 1):
            draw.line([(center_x + offset, top_y), (center_x + offset, bottom_y)], fill=255, width=1)

    # Optional top serif (small horizontal bar)
    if has_top_serif:
        serif_width = random.randint(2, 4)
        draw.line([(center_x - serif_width, top_y), (center_x + serif_width, top_y)], fill=255, width=1)

    # Optional bottom serif
    if has_bottom_serif:
        serif_width = random.randint(2, 4)
        draw.line([(center_x - serif_width, bottom_y), (center_x + serif_width, bottom_y)], fill=255, width=1)

    # Apply slight rotation
    if abs(angle) > 1:
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    # Convert to numpy array
    arr = np.array(img, dtype=np.uint8)

    # Add slight Gaussian noise
    noise = np.random.normal(0, 5, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return arr


def generate_dataset(num_samples):
    """Generate a dataset of synthetic '1' images."""
    images = np.zeros((num_samples, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(num_samples):
        images[i] = generate_one()
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    return images


def save_csv(images, labels, path):
    """Save images and labels to CSV file."""
    print(f"Saving {path}...")
    with open(path, 'w') as f:
        for i in range(len(images)):
            label = labels[i]
            pixels = images[i].flatten()
            line = f"{label}," + ",".join(str(p) for p in pixels)
            f.write(line + "\n")
    print(f"  Saved {len(images)} samples ({path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    print(f"Generating synthetic '1' images...")
    print(f"  Train: {NUM_TRAIN}, Test: {NUM_TEST}")

    random.seed(42)
    np.random.seed(42)

    print("\nGenerating train set...")
    train_images = generate_dataset(NUM_TRAIN)
    train_labels = np.full(NUM_TRAIN, 1, dtype=np.int64)

    print("\nGenerating test set...")
    test_images = generate_dataset(NUM_TEST)
    test_labels = np.full(NUM_TEST, 1, dtype=np.int64)

    print(f"\nPixel stats (train):")
    print(f"  Min: {train_images.min()}, Max: {train_images.max()}, Mean: {train_images.mean():.1f}")

    save_csv(train_images, train_labels, OUTPUT_DIR / "synthetic-1-train.csv")
    save_csv(test_images, test_labels, OUTPUT_DIR / "synthetic-1-test.csv")

    print(f"\nDone. Output files:")
    for f in OUTPUT_DIR.glob("synthetic-1-*.csv"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
