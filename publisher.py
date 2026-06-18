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
import nats

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
IMG_SIZE = 26
BORDER = 1


def postprocess(grid):
    arr = np.array(grid, dtype=np.uint8) * 255
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        is_full_image = (rmin == 0 and rmax == IMG_SIZE - 1 and cmin == 0 and cmax == IMG_SIZE - 1)
        offset_y = (28 - h) // 2 - rmin
        offset_x = (28 - w) // 2 - cmin
        centered = np.zeros((28, 28), dtype=np.uint8)
        sy1, sy2 = max(0, -offset_y), min(IMG_SIZE, IMG_SIZE - offset_y)
        sx1, sx2 = max(0, -offset_x), min(IMG_SIZE, IMG_SIZE - offset_x)
        dy1, dy2 = max(0, offset_y), min(28, 28 + offset_y)
        dx1, dx2 = max(0, offset_x), min(28, 28 + offset_x)
        h_copy = min(sy2 - sy1, dy2 - dy1)
        w_copy = min(sx2 - sx1, dx2 - dx1)
        if h_copy > 0 and w_copy > 0:
            centered[dy1:dy1+h_copy, dx1:dx1+w_copy] = arr[sy1:sy1+h_copy, sx1:sx1+w_copy]
    else:
        centered = np.zeros((28, 28), dtype=np.uint8)
        is_full_image = False

    pil_img = PILImage.fromarray(centered)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(pil_img)
    arr[:BORDER, :] = 0
    arr[-BORDER:, :] = 0
    arr[:, :BORDER] = 0
    arr[:, -BORDER:] = 0
    return arr, not is_full_image


class Line:
    def __init__(self, line_id, image_id):
        self.line_id = line_id
        self.image_id = image_id


class Image:
    def __init__(self, num_lines, thickness, image_id):
        self.num_lines = num_lines
        self.thickness = thickness
        self.image_id = image_id
        self.lines = [Line(line_id, image_id) for line_id in range(1, num_lines + 1)]
        self.grid = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)

        for line_id, line in enumerate(self.lines):
            self.add_line(line_id)

    def add_line(self, line_id):
        temp_grid = np.zeros(self.grid.shape).astype(self.grid.dtype)
        oldgrid = np.copy(self.grid)
        line = self.lines[line_id]
        
        for i in range(10):
            buffer = self.thickness - 1
            y = random.randint(buffer, max(buffer + 1, IMG_SIZE - buffer - 1))
            x = random.randint(buffer, max(buffer + 1, IMG_SIZE - buffer - 1))

            if self.grid[y, x] == 0:
                max_length = max(1, (13 * 13) // (self.thickness * self.thickness))
                num_segments = random.randint(1, max_length + 1)

                ny = y
                nx = x

                for segment_id in range(num_segments):
                    if max_length < 1:
                        break

                    length = random.randint(1, max(1, int(max(max(ny, IMG_SIZE - ny), max(nx, IMG_SIZE - nx)))))
                    slope = random.randint(1, max(1, length // self.thickness + 1)) / length
                    direction = random.randint(0, 7)
                    seg_thickness = self.thickness

                    newlength = 0
                    ty = y
                    tx = x
                    while True:
                        if ty < 0 or ty > IMG_SIZE - 1:
                            break
                        if tx < 0 or tx > IMG_SIZE - 1:
                            break
                        newlength += seg_thickness // 2 + 1

                        if direction == 0:
                            ty = ty + slope
                            tx = tx + seg_thickness // 2 + 1
                        elif direction == 1:
                            ty = ty - seg_thickness // 2 + 1
                            tx = tx + slope
                        elif direction == 2:
                            ty = ty - seg_thickness // 2 + 1
                            tx = tx - slope
                        elif direction == 3:
                            ty = ty - slope
                            tx = tx - seg_thickness // 2 + 1
                        elif direction == 4:
                            ty = ty + slope
                            tx = tx - seg_thickness // 2 + 1
                        elif direction == 5:
                            ty = ty + seg_thickness // 2 + 1
                            tx = tx - slope
                        elif direction == 6:
                            ty = ty + seg_thickness // 2 + 1
                            tx = tx + slope
                        elif direction == 7:
                            ty = ty - slope
                            tx = tx + seg_thickness // 2 + 1

                    length = newlength

                    for step in range(length):
                        if direction == 0:
                            ny = ny + slope
                            nx = nx + seg_thickness // 2 + 1
                        elif direction == 1:
                            ny = ny - seg_thickness // 2 + 1
                            nx = nx + slope
                        elif direction == 2:
                            ny = ny - seg_thickness // 2 + 1
                            nx = nx - slope
                        elif direction == 3:
                            ny = ny - slope
                            nx = nx - seg_thickness // 2 + 1
                        elif direction == 4:
                            ny = ny + slope
                            nx = nx - seg_thickness // 2 + 1
                        elif direction == 5:
                            ny = ny + seg_thickness // 2 + 1
                            nx = nx - slope
                        elif direction == 6:
                            ny = ny + seg_thickness // 2 + 1
                            nx = nx + slope
                        elif direction == 7:
                            ny = ny - slope
                            nx = nx + seg_thickness // 2 + 1

                        dt = seg_thickness // 2
                        if slope < 1 and seg_thickness < 1:
                            dt = 2

                        if np.any(temp_grid[int(ny):int(ny) + 1, int(nx):int(nx) + 1] == 1):
                            max_length = max_length - step
                            break

                        temp_grid[int(ny) - dt:int(ny) + dt + 1, int(nx) - dt:int(nx) + dt + 1] = 1

                    max_length = max_length - length

                    self.grid = np.clip(self.grid + temp_grid, 0, 1)

                break

            if i == 10:
                print("failed adding startpos for line ", line_id)

        newgrid = np.copy(np.clip(self.grid + temp_grid, 0, 1))
        newlines = np.clip(newgrid - oldgrid, 0, 1)
        newlines = ndimage.binary_dilation(newlines)

        newlines_label = label(newlines)
        newlines_label_m = newlines_label == 1
        newlines_label_m2 = newlines_label_m.astype(int)
        newlines_label_m2e = ndimage.binary_erosion(newlines)
        self.grid = np.clip(oldgrid + newlines_label_m2e, 0, 1)


class World:
    def __init__(self):
        pass

    def get_random_image(self):
        num_lines = random.randint(1, 13)
        thickness = random.randint(2, max(2, 13 // num_lines))
        image = Image(num_lines, thickness, 1)

        lines_meta = []
        for line in image.lines:
            line_meta = {
                "line_id": line.line_id,
            }
            lines_meta.append(line_meta)

        assert len(lines_meta) == num_lines, f"Image generation failed: expected {num_lines} lines, got {len(lines_meta)}"

        grid, centered = postprocess(image.grid)
        metadata = {
            "num_lines": len(lines_meta),
            "thickness": image.thickness,
            "centered": centered,
            "lines": lines_meta,
        }
        return grid, metadata


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
