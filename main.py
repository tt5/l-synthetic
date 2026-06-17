#!/usr/bin/env python3
"""generate.py

Generate synthetic shape images with clean separation:
1. Raw shapes (pure geometry)
2. Postprocessing (centering, border, blur)

Usage:
    uv run generate.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

IMG_SIZE = 28
BORDER = 1


# ─── Shape registry ────────────────────────────────────────────────
# Add shapes here: (name, function_that_returns_polygon_points)
# Points should be normalized to roughly (-1, -1) to (1, 1)

RAW_SHAPES = []


# ═══════════════════════════════════════════════════════════════════
# POSTPROCESSING — Centering, border, blur
# ═══════════════════════════════════════════════════════════════════

def center_shape(arr):
    """Center the shape within the image by bounding box."""
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not rows.any():
        return arr
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    offset_y = (IMG_SIZE - h) // 2 - rmin
    offset_x = (IMG_SIZE - w) // 2 - cmin
    centered = np.zeros_like(arr)
    sy1, sy2 = max(0, -offset_y), min(IMG_SIZE, IMG_SIZE - offset_y)
    sx1, sx2 = max(0, -offset_x), min(IMG_SIZE, IMG_SIZE - offset_x)
    dy1, dy2 = max(0, offset_y), min(IMG_SIZE, IMG_SIZE + offset_y)
    dx1, dx2 = max(0, offset_x), min(IMG_SIZE, IMG_SIZE + offset_x)
    h_copy = min(sy2 - sy1, dy2 - dy1)
    w_copy = min(sx2 - sx1, dx2 - dx1)
    if h_copy > 0 and w_copy > 0:
        centered[dy1:dy1+h_copy, dx1:dx1+w_copy] = arr[sy1:sy1+h_copy, sx1:sx1+w_copy]
    return centered


def add_border(arr):
    """Add 1-pixel white border around image."""
    arr[:BORDER, :] = 0
    arr[-BORDER:, :] = 0
    arr[:, :BORDER] = 0
    arr[:, -BORDER:] = 0
    return arr


def apply_blur(arr):
    """Apply gaussian blur."""
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return np.array(img)


def postprocess(img):
    """Full postprocessing pipeline: center → border → blur."""
    arr = center_shape(np.array(img))
    arr = add_border(arr)
    arr = apply_blur(arr)
    return arr


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    path = output_dir / "synthetic.csv"
    print(f"Saving {path}...")
    with open(path, 'w') as f:
        for i in range(len(images)):
            line = f"{labels[i]}," + ",".join(str(p) for p in images[i].flatten())
            f.write(line + "\n")
    print(f"Done. {len(images)} samples.")


if __name__ == "__main__":
    main()
