#!/usr/bin/env python3
"""classifier.py

Subscribe to NATS subject "one", classify images using metadata,
publish labeled images to NATS subject "two".

Usage:
    uv run classifier.py
"""

import asyncio
import json
import numpy as np
import nats
from scipy import ndimage

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT_IN = "one"
SUBJECT_OUT = "two"


def classify(metadata, image):
    """Classify image based on metadata and image pixels. Returns class_id (0-9) or -1."""
    num_lines = metadata["num_lines"]
    thickness = metadata["thickness"]
    lines = metadata["lines"]
    centered = metadata["centered"]

    max_length = max((l["total_length"] for l in lines), default=0)
    total_segments = sum(l["num_segments"] for l in lines)

    num_straight_lines = 0
    num_points = 0
    for line in lines:
        if line["total_length"] >= 3:
            slopes = [s["slope"] for s in line["segments"]]
            if slopes and (max(slopes) - min(slopes)) < 1:
                num_straight_lines += 1
        elif line["total_length"] <= 2:
            num_points += 1

    # Hole detection: check if any background component doesn't touch border
    binary = (image > 0).astype(int)
    inverted = 1 - binary
    labeled, num_features = ndimage.label(inverted)
    has_hole = False
    for i in range(1, num_features + 1):
        component = labeled == i
        touches_border = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )
        if not touches_border:
            has_hole = True
            break

    # Class 0: centered dot
    if num_lines == 1 and max_length <= 2:
        return 0

    # Class 1: 2 or more dots
    if num_lines > 1 and max_length <= 2:
        return 1

    # Class 2: one line
    if num_lines == 1 and max_length > 2:
        return 2

    # Class 3: one line and one or more dots
    if num_lines > 1 and num_straight_lines == 1 and num_points > 0:
        return 3

    # Class 4: two lines
    if num_lines == 2:
        return 4

    # Class 5: hole/ring
    if has_hole:
        return 5

    # Class 6: 3+ straight lines
    if num_straight_lines >= 3:
        return 6

    return -1


async def main():
    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}, listening on '{SUBJECT_IN}'")
    
    async def on_msg(msg):
        data = json.loads(msg.data.decode())
        metadata = data["metadata"]
        
        label = classify(metadata)
        
        output = {
            "frame": data["frame"],
            "image": data["image"],
            "width": data["width"],
            "height": data["height"],
            "metadata": metadata,
            "label": label,
        }
        
        await nc.publish(SUBJECT_OUT, json.dumps(output).encode())
    
    await nc.subscribe(SUBJECT_IN, cb=on_msg)
    
    print("Running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
