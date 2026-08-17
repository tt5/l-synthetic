#!/usr/bin/env python3
"""Upscale images in data/class_0 based on tight bounding boxes.

For each .npy image:
  - compute the tight bounding box of non-zero pixels
  - choose a random scale in [1.0, max_scale]
  - upscale the cropped ROI
  - paste it centered into a target-size canvas

Defaults mirror the repo's 28x28 convention.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


def tight_bbox(grid: np.ndarray):
    rows = np.any(grid > 0, axis=1)
    cols = np.any(grid > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(rmin), int(rmax), int(cmin), int(cmax))


def upscale_image(grid: np.ndarray, scale: float, target_size: int) -> np.ndarray:
    bbox = tight_bbox(grid)
    if bbox is None:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    rmin, rmax, cmin, cmax = bbox
    roi = grid[rmin:rmax + 1, cmin:cmax + 1]

    image_max_scale = min(
        target_size / roi.shape[0],
        target_size / roi.shape[1],
    )
    image_max_scale = max(1.0, image_max_scale)
    scale = min(scale, image_max_scale)

    new_h = max(1, int(round(roi.shape[0] * scale)))
    new_w = max(1, int(round(roi.shape[1] * scale)))
    roi_img = Image.fromarray(roi)
    upscaled = roi_img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

    canvas = Image.new("L", (target_size, target_size), 0)
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    canvas.paste(upscaled, (x, y))
    return np.array(canvas)


def process_class_0(data_dir: str, out_dir: str, target_size: int, seed: int, in_place: bool):
    random.seed(seed)
    np.random.seed(seed)

    class_dir = Path(data_dir) / "class_0"
    if not class_dir.exists():
        raise SystemExit(f"Missing directory: {class_dir}")

    out_class_dir = Path(out_dir) if out_dir else class_dir
    out_class_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(class_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {class_dir}")

    for idx, path in enumerate(files, start=1):
        grid = np.load(path)
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {grid.shape}")

        bbox = tight_bbox(grid)
        if bbox is None:
            new_grid = np.zeros((target_size, target_size), dtype=np.uint8)
        else:
            rmin, rmax, cmin, cmax = bbox
            roi = grid[rmin:rmax + 1, cmin:cmax + 1]
            roi_h, roi_w = roi.shape
            max_scale = min(
                target_size / max(1, roi_h),
                target_size / max(1, roi_w),
            )
            max_scale = max(1.0, max_scale)
            scale = random.uniform(1.0, max_scale)

            new_h = max(1, int(round(roi_h * scale)))
            new_w = max(1, int(round(roi_w * scale)))
            upscaled = Image.fromarray(roi).resize((new_w, new_h), resample=Image.Resampling.NEAREST)

            canvas = Image.new("L", (target_size, target_size), 0)
            x = (target_size - new_w) // 2
            y = (target_size - new_h) // 2
            canvas.paste(upscaled, (x, y))
            new_grid = np.array(canvas)

        if in_place:
            np.save(path, new_grid)
        else:
            out_path = out_class_dir / path.name
            np.save(out_path, new_grid)

        if idx % 100 == 0:
            print(f"{idx}/{len(files)} images processed", flush=True)

    print("Done.")
    print(f"In-place: {in_place}")
    print(f"Output dir: {out_class_dir}")


def main():
    parser = argparse.ArgumentParser(description="Bounding-box based upscaler for class_0")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--out-dir", default="data/class_0_upscaled", help="Output directory, ignored if --in-place")
    parser.add_argument("--target-size", type=int, default=28, help="Canvas size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original files")
    args = parser.parse_args()

    out_dir = None if args.in_place else args.out_dir
    process_class_0(args.data_dir, out_dir, args.target_size, args.seed, args.in_place)


if __name__ == "__main__":
    main()
