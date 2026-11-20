#!/usr/bin/env python3
"""generate_class_4.py

Generate class-4 images: two long random-walk lines that touch or cross,
rendering as a single 8-connected component.

classifier.classify is unchanged and still labels two-long-line images
as class 2 (num_lines >= 2, all lines long with total_length >= 22, no
large white area). The class-4 split is applied here on top: a draw is
accepted only if the foreground is exactly one component under
8-connectivity. Separated draws (>= 2 components) belong to class 2
(generate_class_2.py), so the two classes are disjoint by construction.
Diagonal touches count as connected and land here by design.

World only ever draws 1 or 2 lines, so "two or more long lines" in this
pipeline means exactly 2; this script draws num_lines=2 directly and
reuses src.world.Image's exact rendering pipeline, so output is
pixel-identical in style to generate_dataset.py -- only the 1-vs-2 line
coin flip is bypassed.

Uniqueness is enforced at save time with a hash set seeded from the
images already on disk, so every save is collision-free across runs.

Usage:
    uv run generate_class_4.py [--count N] [--data-dir data] [--seed S]
        [--reject-holes 1]
"""

import argparse
import hashlib
import os

import numpy as np
from scipy import ndimage

from src.world import Image
from src.utils import postprocess
from classifier import classify

DEFAULT_DATA_DIR = "data"
DEFAULT_COUNT = 7000
CLASS_ID = 4
BASE_CLASS = 2  # classifier.py's label for two-long-line images
MAX_HASH_ATTEMPTS = 200

CONNECTIVITY = np.ones((3, 3), dtype=bool)  # 8-connectivity


def count_components(arr):
    """Number of 8-connected foreground components of the rendered image.

    8-connectivity is deliberate: diagonally-touching strokes count as
    one component and land in class 4, not class 2.
    """
    binary = arr > 127
    _, n = ndimage.label(binary, structure=CONNECTIVITY)
    return n


def is_connected(arr):
    """Accept only touching/crossing lines: exactly one component."""
    return count_components(arr) == 1


def has_hole(arr):
    """True if the image encloses a background region not touching the border.

    Two crossing lines can close into a loop, leaving an interior hole.
    Rejecting holes pushes class 4 toward cleaner crossing shapes.
    """
    background = arr <= 127
    labeled, n = ndimage.label(background)
    if n <= 1:
        return False
    border_labels = set(labeled[0, :].tolist()) | set(labeled[-1, :].tolist()) \
        | set(labeled[:, 0].tolist()) | set(labeled[:, -1].tolist())
    border_labels.discard(0)
    holes = [l for l in range(1, n + 1) if l not in border_labels]
    return len(holes) > 0


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
    parser = argparse.ArgumentParser(description="Generate class-4 connected two-line images")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Total target images in class_4 (default: 7000)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Dataset root containing class_4/ (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--reject-holes", type=int, default=1, choices=[0, 1],
                        help="Reject images that enclose a background hole "
                             "(crossings that close into a loop). Default 1 "
                             "(reject). Set 0 to allow holes.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    reject_holes = bool(args.reject_holes)

    class_dir = os.path.join(args.data_dir, f"class_{CLASS_ID}")
    os.makedirs(class_dir, exist_ok=True)

    existing, next_index, seen = scan_class_dir(class_dir)
    target_new = max(0, args.count - existing)
    print(f"Generating in {class_dir}: {existing} existing, target {args.count} "
          f"total -> {target_new} new images")
    print("Accepting connected two-line draws: exactly 1 component "
          "(8-connectivity)")

    written = 0
    rejected = 0
    separated = 0
    holed = 0

    def accept(arr):
        """Full acceptance test for a rendered draw."""
        if not is_connected(arr):
            return "separated"
        if reject_holes and has_hole(arr):
            return "hole"
        return "ok"

    while written < target_new:
        # draw num_lines=2 directly; classify still calls these class 2,
        # the single-component filter below is what makes them class 4
        grid, metadata = build_image(num_lines=2, thickness=1)
        arr, metadata["centered"] = postprocess(grid)
        if classify(metadata, arr) != BASE_CLASS:
            rejected += 1
            if rejected % 500 == 0:
                print(f"  {rejected} draws rejected (not two long lines), "
                      f"{written}/{target_new} written", flush=True)
            continue
        # accept only touching/crossing lines; separated draws are
        # class-2 shapes and are rejected here
        verdict = accept(arr)
        if verdict == "separated":
            separated += 1
            if separated % 500 == 0:
                print(f"  {separated} separated draws rejected "
                      f"(-> class 2), {written}/{target_new} written",
                      flush=True)
            continue
        if verdict == "hole":
            holed += 1
            if holed % 500 == 0:
                print(f"  {holed} hole draws rejected, "
                      f"{written}/{target_new} written", flush=True)
            continue

        # enforce uniqueness at save time; a colliding or invalid redraw
        # is discarded, never saved
        attempts = 0
        while True:
            h = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
            if h not in seen:
                break
            attempts += 1
            if attempts >= MAX_HASH_ATTEMPTS:
                h = None
                break
            # regenerate the whole image on collision (fresh entropy)
            grid, metadata = build_image(num_lines=2, thickness=1)
            arr, metadata["centered"] = postprocess(grid)
            if classify(metadata, arr) != BASE_CLASS or accept(arr) != "ok":
                h = None
                break
        if h is None:
            continue
        seen.add(h)
        np.save(os.path.join(class_dir, f"{next_index + written:06d}.npy"), arr)
        written += 1
        if written % 100 == 0:
            print(f"  {written}/{target_new} written", flush=True)

    print(f"Done. Wrote {written} new images to {class_dir} "
          f"(total {existing + written}).")
    print(f"Rejected {rejected} non-two-line draws, {separated} separated "
          f"(class-2 shapes), {holed} with holes.")


if __name__ == "__main__":
    main()
