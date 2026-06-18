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

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT_IN = "one"
SUBJECT_OUT = "two"


def classify(metadata):
    """Classify image based on metadata. Returns class_id (0-9) or -1."""
    num_lines = metadata["num_lines"]
    thickness = metadata["thickness"]
    lines = metadata["lines"]
    total_length = sum(l["total_length"] for l in lines)
    total_segments = sum(l["num_segments"] for l in lines)
    
    # Class 0: centered dot
    if num_lines == 1 and total_segments < 2 and total_length//thickness <= 3:
        return 0
    
    # Class 1: 2+ not centered dots (multiple short segments)
    if num_lines >= 2 and all(
        l["num_segments"] == 1 and l["segments"][0]["length"] <= thickness * 2
        for l in lines
    ):
        return 1
    
    # Class 2: one line (1 line, any segments)
    if num_lines == 1:
        return 2
    
    # Class 3: one line + dots (2+ lines, one is long, others are short)
    if num_lines >= 2:
        long_lines = [l for l in lines if l["total_length"] > thickness * 3]
        short_lines = [l for l in lines if l["total_length"] <= thickness * 3]
        if len(long_lines) == 1 and len(short_lines) >= 1:
            return 3
    
    # Class 4: two lines, orthogonal, crossing or not
    if num_lines == 2:
        return 4
    
    # Class 5: ring/hole (1 line, many segments forming closed shape)
    if num_lines == 1 and total_segments >= 4:
        return 5
    
    # Class 6: 3+ lines, not regular
    if num_lines >= 3:
        return 6
    
    # Class 7-9: complex structures (placeholder)
    if num_lines >= 4:
        return 7
    
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
