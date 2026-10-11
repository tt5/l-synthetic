#!/usr/bin/env python3
"""generate_class_2.py

Generate class-2 images: two long random-walk lines that cross/overlap.

classifier.classify defines class 2 as num_lines == 2 with both lines
long (total_length >= 22) and no large white area. This script draws
num_lines=2 directly (skipping the 5/6 single-line draws the general
generator wastes) and reuses src.world.Image's exact rendering pipeline,
so output is pixel-identical in style to generate_dataset.py -- only the
1-vs-2 line coin flip is bypassed.

Uniqueness is enforced at save time with a hash set seeded from the
images already on disk, so every save is collision-free across runs.

Usage:
    uv run generate_class_2.py [--count N] [--data-dir data] [--seed S]
"""

import argparse
import hashlib
import os

import numpy as np

from src.world import Image
from src.utils import postprocess
from classifier import classify

DEFAULT_DATA_DIR = "data"
DEFAULT_COUNT = 7000
CLASS_ID = 2
MAX_HASH_ATTEMPTS = 200
DRAW_ATTEMPTS = 100


def build_image(num_lines=2, thickness=1):
    """Draw a num_lines image via src.world.Image and return (grid, metadata)
    matching src.world.World.get_random_image (before postprocess)."""
    image = Image(num_lines, thickness, 1)

    lines_meta = []
    for line in image.lines:
        total_length = sum(s["length"] for s in line.segments)
        lines_meta.append({
            "line_id": line.line_id,
            "num_segments": len(line.segments),
            "total_length": total_length,
            "segments": line.segments,
        })

    assert len(lines_meta) == num_lines, (
        f"Image generation failed: expected {num_lines} lines, "
        f"got {len(lines_meta)}"
    )

    metadata = {
        "num_lines": len(lines_meta),
        "thickness": image.thickness,
        "centered": None,
        "lines": lines_meta,
    }
    return image.grid, metadata


def scan_class_dir(class_dir):
    """Return (existing_count, next_index, hashes of existing images)."""
    hashes = set()
    max_index = -1
    count = 0
    if os.path.isdir(class_dir):
        for f in os.listdir(class_dir):
            if not f.endswith(".npy"):
                continue
            count += 1
            stem = f[:-4]
            if stem.isdigit():
                max_index = max(max_index, int(stem))
            with open(os.path.join(class_dir, f), "rb") as fh:
                hashes.add(hashlib.blake2b(fh.read(), digest_size=16).hexdigest())
    return count, max_index + 1, hashes


def main():
    parser = argparse.ArgumentParser(description="Generate class-2 two-long-line images")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Total target images in class_2 (default: 7000)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Dataset root containing class_2/ (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    class_dir = os.path.join(args.data_dir, f"class_{CLASS_ID}")
    os.makedirs(class_dir, exist_ok=True)

    existing, next_index, seen = scan_class_dir(class_dir)
    target_new = max(0, args.count - existing)
    print(f"Generating in {class_dir}: {existing} existing, target {args.count} "
          f"total -> {target_new} new images")

    written = 0
    rejected = 0
    while written < target_new:
        # draw num_lines=2 directly, then verify it classifies as class 2
        grid, metadata = build_image(num_lines=2, thickness=1)
        arr, metadata["centered"] = postprocess(grid)
        if classify(metadata, arr) != CLASS_ID:
            rejected += 1
            if rejected % 500 == 0:
                print(f"  {rejected} draws rejected (not class 2), "
                      f"{written}/{target_new} written", flush=True)
            continue

        # enforce uniqueness at save time
        attempts = 0
        while True:
            h = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
            if h not in seen or attempts >= MAX_HASH_ATTEMPTS:
                break
            attempts += 1
            # regenerate the whole image on collision (fresh entropy)
            grid, metadata = build_image(num_lines=2, thickness=1)
            arr, metadata["centered"] = postprocess(grid)
            if classify(metadata, arr) != CLASS_ID:
                break
        seen.add(h)
        np.save(os.path.join(class_dir, f"{next_index + written:06d}.npy"), arr)
        written += 1
        if written % 100 == 0:
            print(f"  {written}/{target_new} written", flush=True)

    print(f"Done. Wrote {written} new images to {class_dir} "
          f"(total {existing + written}). Rejected {rejected} non-class-2 draws.")


if __name__ == "__main__":
    main()
