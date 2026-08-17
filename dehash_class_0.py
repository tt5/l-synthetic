#!/usr/bin/env python3
"""Break exact duplicates for centered class_0 discs without changing blur style.

Only applies the duplicate-breaking transform to images that have at least one
exact duplicate. Unique images are copied through unchanged.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def deterministic_rng_for(name: str, seed: int = 42):
    h = hashlib.blake2b(name.encode(), digest_size=8).hexdigest()
    state = (seed + int(h[:16], 16)) & 0xFFFFFFFF
    rng = np.random.RandomState(state)
    return rng


def tight_bbox(arr: np.ndarray):
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(rmin), int(rmax), int(cmin), int(cmax))


def make_unique_disc(grid: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # 1) float 0..1
    arr = np.array(grid, dtype=np.float32) / 255.0

    # 2) binary mask
    mask = arr > 0.5

    # 3) dilate
    mask = ndimage.binary_dilation(mask, iterations=1)

    # 4) perimeter pixels
    shifted = [
        np.roll(mask, 1, axis=0),
        np.roll(mask, -1, axis=0),
        np.roll(mask, 1, axis=1),
        np.roll(mask, -1, axis=1),
        np.roll(np.roll(mask, 1, axis=0), 1, axis=1),
        np.roll(np.roll(mask, 1, axis=0), -1, axis=1),
        np.roll(np.roll(mask, -1, axis=0), 1, axis=1),
        np.roll(np.roll(mask, -1, axis=0), -1, axis=1),
    ]
    perimeter = np.zeros_like(mask, dtype=bool)
    for s in shifted:
        perimeter = np.logical_or(perimeter, np.logical_xor(mask, s))
    perimeter = np.logical_and(perimeter, mask)

    perim_coords = np.argwhere(perimeter)
    if perim_coords.shape[0] == 0:
        target = np.zeros((28, 28), dtype=np.float32)
    else:
        n = 1 if perim_coords.shape[0] == 1 else rng.randint(1, min(4, perim_coords.shape[0]) + 1)
        idx = rng.choice(perim_coords.shape[0], size=min(n, perim_coords.shape[0]), replace=False)
        coords = perim_coords[idx]
        mod = mask.copy()
        mod[coords[:, 0], coords[:, 1]] = False

        rows = np.any(mod, axis=1)
        cols = np.any(mod, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            h = rmax - rmin + 1
            w = cmax - cmin + 1
            oy = (28 - h) // 2 - rmin
            ox = (28 - w) // 2 - cmin
            target = np.zeros((28, 28), dtype=np.float32)
            sy1, sy2 = max(0, -oy), min(27, 27 - oy)
            sx1, sx2 = max(0, -ox), min(27, 27 - ox)
            dy1, dy2 = max(0, oy), min(28, 28 + oy)
            dx1, dx2 = max(0, ox), min(28, 28 + ox)
            hc = min(sy2 - sy1, dy2 - dy1)
            wc = min(sx2 - sx1, dx2 - dx1)
            if hc > 0 and wc > 0:
                target[dy1:dy1+hc, dx1:dx1+wc] = mod[sy1:sy1+hc, sx1:sx1+wc]
        else:
            target = np.zeros((28, 28), dtype=np.float32)

    # 5) re-blur
    pil_img = Image.fromarray((target * 255).astype(np.uint8))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    out = np.array(pil_img, dtype=np.uint8)

    # 6) zero border
    out[:1, :] = 0
    out[-1:, :] = 0
    out[:, :1] = 0
    out[:, -1:] = 0
    return out


def main():
    parser = argparse.ArgumentParser(description="Dehash class_0 centered discs")
    parser.add_argument("--data-dir", default="/home/n/data/l/synthetic/data/class_0", help="Input class_0 directory")
    parser.add_argument("--out-dir", default="/home/n/data/l/synthetic/data/class_0_unique", help="Output directory, ignored if --in-place")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original files")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    base = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(base.glob("*.npy"))
    hashes = {}
    order = []
    for path in files:
        h = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        hashes.setdefault(h, []).append(path)
        order.append(path)

    written = 0
    for i, path in enumerate(order, start=1):
        h = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        group = hashes[h]
        arr = np.load(path)

        if len(group) <= 1:
            out_path = path
            out_arr = arr
        else:
            idx_in_group = group.index(path)
            rng_name = f"{path.stem}_{idx_in_group}"
            out_arr = make_unique_disc(arr, deterministic_rng_for(rng_name, args.seed))
            out_path = path

        np.save(out_path, out_arr)
        written += 1
        if i % 100 == 0:
            print(f"{i}/{len(files)} processed", flush=True)

    total_unique_inputs = len({hashlib.blake2b(p.read_bytes(), digest_size=16).hexdigest() for p in order})
    print(f"scanned= {len(files)}")
    print(f"written= {written}")
    print(f"unique input images= {total_unique_inputs}")


if __name__ == "__main__":
    main()
