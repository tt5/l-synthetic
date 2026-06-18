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
    straight_lines_slopes_max = []
    straight_lines_slopes_min = []
    for line in lines:
        if line["total_length"] >= 3:
            slopes = [s["slope"] for s in line["segments"]]
            straight_lines_slopes_max.append(max(slopes))
            straight_lines_slopes_min.append(min(slopes))
            #print(max(slopes), min(slopes))
            slopes_max_diff = (max(slopes) - min(slopes))
            if slopes and slopes_max_diff < 0.2:
                num_straight_lines += 1
        elif line["total_length"] <= 2:
            num_points += 1

    binary = (image == 0).astype(int)
    labeled, num_features = ndimage.label(binary)

    has_hole = False
    # Check if any black component doesn't touch the border
    for i in range(1, num_features + 1):
        component = labeled == i
        if np.sum(component) < 9:
            continue
        touches_border = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )
        if not touches_border:
            has_hole = True
            break  # Found a hole

    # Label connected components of white pixels
    binary = (image > 127).astype(int)
    labeled, num_features = ndimage.label(binary)

    # Calculate sizes of all white components
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

    # Check if any white component is "large" (e.g., > 20% of image)
    image_size = image.shape[0] * image.shape[1]
    large_threshold = image_size * 0.2
    has_large_white_area = np.any(sizes > large_threshold)


    # Class 0: centered dot
    if num_points == 1 and num_straight_lines == 0:
        return 0

    # Class 1: 2 or more dots
    if num_points > 1 and num_straight_lines == 0:
        return 1

    # Class 2: one line
    if num_points == 0 and num_straight_lines == 1:
        return 2

    # Class 3: one line and one or more dots
    if num_points > 0 and num_straight_lines == 1:
        return 3

    #4. two lines, orthogonal, crossing and not crossing
    if num_straight_lines == 2 and (max(straight_lines_slopes_max) - min(straight_lines_slopes_min)) > 0.3:
        return 4
     
    # Class 6: one hole/ring in the image
    if has_hole:
        return 5
     
    # Class 6. more than 2 clear lines
    if num_points <= 1 and num_straight_lines > 2:
        return 6
     
    # Class 7: complex structure, large white area
    if num_points > 1 and has_large_white_area:
        return 7

    # Class 8: complex structure, very regular, no large white area
    if num_points == 0 and not has_large_white_area:
        return 8
     
    # Class 9: complex structure, very assymetric, no large white area
    if num_points > 1 and num_straight_lines > 2 and not has_large_white_area:
        return 9


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
