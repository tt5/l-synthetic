#!/usr/bin/env python3
"""listener.py

Subscribe to NATS subject "two", display labeled images with ffplay.
Label text is drawn next to the image.

Usage:
    uv run listener.py
"""

import asyncio
import json
import numpy as np
import subprocess
import nats
from PIL import Image as PILImage, ImageDraw, ImageFont

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "two"
FPS = 2
IMG_SIZE = 28
INFO_HEIGHT = 60
INFO_WIDTH = 70


def draw_label(img_array, label, metadata):
    """Draw label and metadata text next to the image. Returns combined image."""
    h, w = img_array.shape
    total_h = max(h, INFO_HEIGHT)
    combined = np.zeros((total_h, INFO_WIDTH + w), dtype=np.uint8)
    pil_img = PILImage.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        font = ImageFont.load_default()

    max_length = max((l["total_length"] for l in metadata["lines"]), default=0)

    lines = [
        f"label: {label}",
        f"lines: {metadata['num_lines']}",
        f"thick: {metadata['thickness']}",
        f"center: {metadata['centered']}",
        f"max_len: {max_length}",
    ]

    y = 2
    for text in lines:
        draw.text((2, y), text, fill=255, font=font)
        y += 10

    combined = np.array(pil_img)
    combined[:h, INFO_WIDTH:] = img_array
    return combined


async def main():
    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}, listening on '{SUBJECT}'")

    total_width = INFO_WIDTH + IMG_SIZE
    total_height = max(IMG_SIZE, INFO_HEIGHT)
    cmd = [
        "ffplay",
        "-f", "rawvideo",
        "-pixel_format", "gray",
        "-video_size", f"{total_width}x{total_height}",
        "-framerate", str(FPS),
        "-",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_count = 0

    async def on_msg(msg):
        nonlocal frame_count
        data = json.loads(msg.data.decode())

        img = np.array(data["image"], dtype=np.uint8).reshape(data["height"], data["width"])
        label = data["label"]
        frame_count += 1

        metadata = data["metadata"]
        combined = draw_label(img, label, metadata)
        proc.stdin.write(combined.tobytes())
        proc.stdin.flush()

        print(f"Frame {frame_count}: label={label}")

    await nc.subscribe(SUBJECT, cb=on_msg)

    print("Running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        proc.stdin.close()
        proc.wait()
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
