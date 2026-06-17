#!/usr/bin/env python3
"""listener.py

Subscribe to NATS subject "two", display labeled images with ffplay.

Usage:
    uv run listener.py
"""

import asyncio
import json
import numpy as np
import subprocess
import nats

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "two"
FPS = 2


async def main():
    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}, listening on '{SUBJECT}'")
    
    # Start ffplay
    cmd = [
        "ffplay",
        "-f", "rawvideo",
        "-pixel_format", "gray",
        "-video_size", "28x28",
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
        
        print(f"Frame {frame_count}: label={label}")
        
        proc.stdin.write(img.tobytes())
        proc.stdin.flush()
    
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
