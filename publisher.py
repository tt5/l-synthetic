#!/usr/bin/env python3
"""publisher.py

Generate random pictogram images using the World class from src/world.py.
Publishes to NATS subject "one" with image + metadata.

Usage:
    uv run publisher.py
"""

import asyncio
import json

from src.world import World
import nats

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2


async def main():
    nc = await nats.connect(NATS_URL)
    world = World()
    print(f"Connected to {NATS_URL}, publishing to '{SUBJECT}' at {FPS} fps")

    frame = 0
    try:
        while True:
            grid, metadata = world.get_random_image()

            message = {
                "frame": frame,
                "image": grid.tolist(),
                "width": 28,
                "height": 28,
                "metadata": metadata,
            }

            await nc.publish(SUBJECT, json.dumps(message).encode())
            frame += 1
            await asyncio.sleep(1.0 / FPS)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
