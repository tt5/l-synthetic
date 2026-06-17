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
LABEL_WIDTH = 40


def draw_label(img_array, label):
    """Draw label text next to the image. Returns combined image."""
    h, w = img_array.shape
    
    # Create combined image: label area + original image
    combined = np.zeros((h, LABEL_WIDTH + w), dtype=np.uint8)
    
    # Draw label text on the left side
    pil_img = PILImage.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        font = ImageFont.load_default()
    
    text = str(label)
    # Center text vertically in the label area
    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    text_w = bbox[2] - bbox[0]
    y = (h - text_h) // 2
    x = (LABEL_WIDTH - text_w) // 2
    draw.text((x, y), text, fill=255, font=font)
    
    combined = np.array(pil_img)
    
    # Copy original image to the right side
    combined[:, LABEL_WIDTH:] = img_array
    
    return combined


async def main():
    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}, listening on '{SUBJECT}'")

    total_width = LABEL_WIDTH + IMG_SIZE
    cmd = [
        "ffplay",
        "-f", "rawvideo",
        "-pixel_format", "gray",
        "-video_size", f"{total_width}x{IMG_SIZE}",
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

        combined = draw_label(img, label)
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
