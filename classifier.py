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
    """Classify image based on metadata and image pixels.

    Only classes 1 (one long line) and 2 (two or more long lines) are
    produced; class 0 comes from generate_class_0.py. Everything else
    returns -1 (skipped, not saved as a class).
    """
    num_lines = metadata["num_lines"]
    lines = metadata["lines"]

    num_long_lines = 0
    num_points = 0
    for line in lines:
        if line["total_length"] <= 7:
            num_points += 1
        elif line["total_length"] >= 22:
            num_long_lines += 1

    # Label connected components of white pixels
    binary = (image > 127).astype(int)
    labeled, num_features = ndimage.label(binary)

    # Calculate sizes of all white components
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

    # Check if any white component is "large" (e.g., > 20% of image)
    image_size = image.shape[0] * image.shape[1]
    large_threshold = image_size * 0.2
    has_large_white_area = np.any(sizes > large_threshold)

    # Class 1: one long line
    if num_points == 0 and num_long_lines == 1 and not has_large_white_area:
        return 1

    # Class 2: two or more long lines
    if num_points == 0 and num_long_lines >= 2 and not has_large_white_area:
        return 2

    return -1


async def main():
    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}, listening on '{SUBJECT_IN}'")
    
    async def on_msg(msg):
        data = json.loads(msg.data.decode())
        metadata = data["metadata"]
        
        image = np.array(data["image"], dtype=np.uint8).reshape(data["height"], data["width"])
        label = classify(metadata, image)
        
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
