#!/usr/bin/env python3
"""generate.py

Generate synthetic shape images with clean separation:
1. Raw shapes (pure geometry)
2. Parameters (thickness, length, orientation, variation)
3. Postprocessing (centering, border, blur)

Usage:
    uv run generate.py --num-shapes 10 --count 6000
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import random
import math

IMG_SIZE = 28
BORDER = 1


# ─── Shape registry ────────────────────────────────────────────────
# Add shapes here: (name, function_that_returns_polygon_points)
# Points should be normalized to roughly (-1, -1) to (1, 1)

RAW_SHAPES = [
    # Example: ("circle", lambda: [(math.cos(a), math.sin(a)) for a in [math.radians(i*18) for i in range(20)]]),
]


# ═══════════════════════════════════════════════════════════════════
# 2. PARAMETERS — Thickness, length, orientation, variation
# Transforms raw shape points into final shape on canvas.
# ═══════════════════════════════════════════════════════════════════

def get_params(class_id, total_classes):
    """Generate parameters for a given class.
    
    Returns dict with:
        shape_idx: which base shape
        size: scale factor (length)
        thickness: line thickness
        angle: rotation in degrees
        variation: variant index
    """
    rng = random.Random(class_id * 9973)
    return {
        "shape_idx": class_id % len(RAW_SHAPES),
        "size": rng.randint(4, 10),
        "thickness": rng.randint(1, 3),
        "angle": rng.uniform(0, 360),
        "variation": class_id // len(RAW_SHAPES),
    }


def apply_params(draw, params, cx, cy):
    """Draw a shape with given parameters at position (cx, cy)."""
    shape_name, shape_fn = RAW_SHAPES[params["shape_idx"]]
    raw_pts = shape_fn()
    
    # Scale by size
    size = params["size"]
    pts = [(cx + p[0] * size, cy + p[1] * size) for p in raw_pts]
    
    # Apply rotation
    angle = math.radians(params["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = []
    for px, py in pts:
        dx, dy = px - cx, py - cy
        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        rotated.append((rx, ry))
    
    # Draw filled polygon
    if len(rotated) > 2:
        draw.polygon(rotated, fill=255)
    elif len(rotated) == 2:
        draw.line(rotated, fill=255, width=params["thickness"])


# ═══════════════════════════════════════════════════════════════════
# 3. POSTPROCESSING — Centering, border, blur
# Applied after shape is drawn.
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
    """Add 2-pixel white border around image."""
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
# IMAGE GENERATION — Combines shape + params + postprocessing
# ═══════════════════════════════════════════════════════════════════

def generate_image(class_id, total_classes, noise_level=5):
    """Generate a single synthetic image for the given class."""
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    
    params = get_params(class_id, total_classes)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    
    apply_params(draw, params, cx, cy)
    
    arr = postprocess(img)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return arr


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--num-shapes", type=int, required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--noise", type=float, default=5)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = args.num_shapes * args.count
    print(f"Generating {total} images ({args.num_shapes} classes × {args.count})...")
    
    images, labels = [], []
    for class_id in range(args.num_shapes):
        for _ in range(args.count):
            images.append(generate_image(class_id, args.num_shapes, args.noise))
            labels.append(class_id)
        if (class_id + 1) % 10 == 0 or class_id == args.num_shapes - 1:
            print(f"  Class {class_id + 1}/{args.num_shapes}")
    
    path = output_dir / f"synthetic-{args.num_shapes}classes.csv"
    print(f"Saving {path}...")
    with open(path, 'w') as f:
        for i in range(len(images)):
            line = f"{labels[i]}," + ",".join(str(p) for p in images[i].flatten())
            f.write(line + "\n")
    print(f"Done. {len(images)} samples, {args.num_shapes} classes.")


if __name__ == "__main__":
    main()
