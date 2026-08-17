#!/usr/bin/env python3
"""publisher.py

Generate random pictogram images using the same World class as main.py.
Publishes to NATS subject "one" with image + metadata.

Usage:
    uv run publisher.py
"""

import asyncio
import json
import random
import numpy as np
from PIL import Image as PILImage, ImageFilter
from scipy import ndimage
from skimage.measure import label
from src.utils import postprocess
import nats
NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
IMG_SIZE = 26
BORDER = 1


class Line:
    def __init__(self, line_id, image_id):
        self.line_id = line_id
        self.image_id = image_id
        self.segments = []


class Image:
    def __init__(self, num_lines, thickness, image_id):
        self.num_lines = num_lines
        self.thickness = thickness
        self.image_id = image_id

        # each Line object starts empty (no segments)
        self.lines = [Line(line_id, image_id) for line_id in range(1, num_lines + 1)]
        self.grid = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)

        for line_id, line in enumerate(self.lines):
            self.add_line(line_id)

    def add_line(self, line_id):
        temp_grid = np.zeros(self.grid.shape).astype(self.grid.dtype)
        line = self.lines[line_id]

        while True:
            img = np.zeros((26,26), dtype="float32")
            x = np.random.randint(0,25)
            y = np.random.randint(0,13)
            direction = np.random.randint(4, 8)

            if direction == 4:
                slope = np.random.randint(0,44)/45
            elif direction == 5:
                slope = np.random.randint(0, 44)/45
            elif direction == 6:
                slope = np.random.randint(0,44)/45
            elif direction == 7:
                slope = np.random.randint(0,44)/45
                
            length = np.random.randint(1,26)

            nx = x
            ny = y
            seglength = 1
            step = 0
            segmentid = 0
            segments = []

            while length > 0:
                length -= 1
                step += 1
                
                if direction == 0:
                    ny = ny + slope
                    nx = nx + 1
                if direction == 1:
                    ny = ny - 1
                    nx = nx + slope
                if direction == 2:
                    ny = ny - 1
                    nx = nx - slope
                if direction == 3:
                    ny = ny - slope
                    nx = nx - 1
                if direction == 4:
                    ny = ny + slope
                    nx = nx - 1
                if direction == 5:
                    ny = ny + 1
                    nx = nx - slope
                if direction == 6:
                    ny = ny + 1
                    nx = nx + slope
                if direction == 7:
                    ny = ny + slope
                    nx = nx + 1
                    
                if 25 < nx or nx < 0 or 25 < ny or ny < 0:
                    break

                if step > 1:
                    img[int(ny), int(nx)] = segmentid
                else:
                    img[int(ny), int(nx)] = 0

                if seglength == 0:
                    dirchange = np.random.randint(-1,2)
                    if direction == 0 and dirchange == -1:
                        direction = 7
                    if direction == 7 and dirchange == 1:
                        direction = 0
                    else:
                        direction += dirchange
                    slope = np.random.randint(0,44)/45

                    seglength = np.random.randint(2,13)
                    length = seglength
                    segmentid += 1
                    segments.append(length)
                seglength -= 1
                
            if np.all(img == 0):
                continue
            else:
                break

        rot = np.random.randint(0,4)
        if rot == 1:
            img = np.rot90(img)
        elif rot == 2:
            img = np.rot90(np.rot90(img))
        elif rot == 3:
            img = np.rot90(np.rot90(np.rot90((img))))

        img = np.clip(img, 0, 1)
        img = ndimage.binary_dilation(img)

        arr = np.array(img, dtype=np.uint8) * 255
        pil_img = PILImage.fromarray(arr)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.57))
        img = np.array(pil_img)
        img = (img > 195).astype(int)

        rows = np.any(arr > 0, axis=1)
        cols = np.any(arr > 0, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

        if np.random.randint(0,2) == 0:
            roll = np.random.randint(0,rmin+1)
            img = np.roll(img, -roll, 0)
        else:
            roll = np.random.randint(0, 26-rmax)
            img = np.roll(img, roll, 0)

        if np.random.randint(0,2) == 0:
            roll = np.random.randint(0,cmin+1)
            img = np.roll(img, -roll, 1)
        else:
            roll = np.random.randint(0, 26-cmax)
            img = np.roll(img, roll, 1)

        for s_id, length in enumerate(segments):
            line.segments.append({
                "segment_id": s_id,
                "length": length,
                "slope": 0,
                "direction": 0,
            })

        self.grid = self.grid + img


class World:
    def __init__(self):
        pass

    def get_random_image(self):
        if random.randint(0,5) == 0:
            #num_lines = random.randint(2, 13)
            num_lines = 2
        else:
            num_lines = 1
        thickness = 1
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

        assert len(lines_meta) == num_lines, f"Image generation failed: expected {num_lines} lines, got {len(lines_meta)}"

        metadata = {
            "num_lines": len(lines_meta),
            "thickness": image.thickness,
            "centered": None,  # will be set after postprocessing
            "lines": lines_meta,
        }
        return image.grid, metadata


async def main():
    nc = await nats.connect(NATS_URL)
    world = World()
    print(f"Connected to {NATS_URL}, publishing to '{SUBJECT}' at {FPS} fps")

    frame = 0
    try:
        while True:
            grid, metadata = world.get_random_image()
            grid, metadata["centered"] = postprocess(grid)

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
