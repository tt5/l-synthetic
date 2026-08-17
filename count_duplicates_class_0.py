#!/usr/bin/env python3
"""Find duplicate .npy arrays in data/class_0."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    base = Path("/home/n/data/l/synthetic/data/class_0")
    files = sorted(base.glob("*.npy"))
    if not files:
        print(f"No .npy files in {base}")
        return

    def hash_arr(path):
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"Failed loading {path}: {e}")
            return None
        return (arr.shape, arr.tobytes())

    counts = Counter()
    groups = defaultdict(list)
    skip = 0

    for path in files:
        h = hash_arr(path)
        if h is None:
            skip += 1
            continue
        counts[h] += 1
        groups[h].append(path.name)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    total_duplicates = sum(len(v) - 1 for v in dup_groups.values())
    max_group = max(dup_groups.values(), key=len) if dup_groups else []

    print(f"Files scanned : {len(files)}")
    print(f"Unreadable    : {skip}")
    print(f"Unique images : {len(groups)}")
    print(f"Duplicate groups: {len(dup_groups)}")
    print(f"Extra copies (duplicates): {total_duplicates}")
    if max_group:
        print(f"Largest group size: {len(max_group)}")


if __name__ == "__main__":
    main()
