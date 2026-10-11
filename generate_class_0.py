#!/usr/bin/env python3
"""generate_class_0.py

Generate class-0 "point" images: disk-like blobs ranging from a small
point to a blob that fills the whole 26x26 canvas, with irregular
(blob-like) borders.

Unlike generate_dataset.py this script does not classify -- every image
is class 0 by construction. Uniqueness is enforced at generation time:

  * blob parameters are continuous (sub-pixel center, log-uniform
    radius, random harmonic amplitude/phase for the border), so the
    reachable image space is effectively infinite;
  * a hash set seeded with the images already on disk makes every save
    collision-free, including across runs.

Usage:
    uv run generate_class_0.py [--count N] [--data-dir data] [--seed S]
"""

import argparse
import hashlib
import os

import numpy as np
from PIL import Image, ImageFilter

IMG_SIZE = 26
DEFAULT_DATA_DIR = "data"
DEFAULT_COUNT = 7000
MAX_HASH_ATTEMPTS = 200


def make_blob(rng):
    """Render one blob into a 26x26 {0,1} mask.

    Shape:  r(theta) = r_mean * (1 + amp*sin(k1*theta+phi1)
                                    + 0.5*amp*sin(k2*theta+phi2))
    The blob is centered in the 26x26 canvas with a small sub-pixel
    jitter; r_mean is log-uniform from 1 (small point) up to 13.5
    (fills the whole canvas), amp in [0, 0.3] (0 is a regular disc,
    0.3 is clearly blob-like) and 1-2 random harmonics. Large blobs
    may clip at the canvas edge.
    """
    cy = IMG_SIZE / 2.0 + rng.uniform(-1.0, 1.0)
    cx = IMG_SIZE / 2.0 + rng.uniform(-1.0, 1.0)
    r_mean = float(np.exp(rng.uniform(np.log(1.0), np.log(13.5))))
    amp = rng.uniform(0.0, 0.3)
    k1 = int(rng.integers(1, 4))
    k2 = int(rng.integers(0, 4))
    phi1 = rng.uniform(0.0, 2.0 * np.pi)
    phi2 = rng.uniform(0.0, 2.0 * np.pi)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE] + 0.5
    theta = np.arctan2(yy - cy, xx - cx)
    dist = np.hypot(xx - cx, yy - cy)
    r = r_mean * (
        1.0
        + amp * np.sin(k1 * theta + phi1)
        + 0.5 * amp * np.sin(k2 * theta + phi2)
    )
    return (dist < r).astype(np.uint8)


def render_28(mask):
    """Blur the 26x26 mask and embed it in a 28x28 canvas with a 1px
    zero border (blob centered, no re-cropping)."""
    pil_img = Image.fromarray(mask * 255)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[1:27, 1:27] = np.array(pil_img, dtype=np.uint8)
    return arr


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
    parser = argparse.ArgumentParser(description="Generate class-0 point/blob images")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Total target images in class_0 (default: 7000)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Dataset root containing class_0/ (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    class_dir = os.path.join(args.data_dir, "class_0")
    os.makedirs(class_dir, exist_ok=True)

    existing, next_index, seen = scan_class_dir(class_dir)
    target_new = max(0, args.count - existing)
    print(f"Generating in {class_dir}: {existing} existing, target {args.count} "
          f"total -> {target_new} new images")

    written = 0
    for k in range(target_new):
        attempts = 0
        while True:
            arr = render_28(make_blob(rng))
            h = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
            if h not in seen:
                break
            attempts += 1
            if attempts >= MAX_HASH_ATTEMPTS:
                print(f"  warning: no unique image after {attempts} attempts, saving anyway")
                break
        seen.add(h)
        np.save(os.path.join(class_dir, f"{next_index + k:06d}.npy"), arr)
        written += 1
        if written % 100 == 0:
            print(f"  {written}/{target_new} written", flush=True)

    print(f"Done. Wrote {written} new images to {class_dir} "
          f"(total {existing + written}).")


if __name__ == "__main__":
    main()
