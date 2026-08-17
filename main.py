#!/usr/bin/env python3

import random
import numpy as np
import imageio
from skimage.measure import label
from PIL import Image as PILImage, ImageFilter
from scipy import ndimage


def postprocess(grid):
    arr = np.array(grid, dtype=np.uint8) * 255

    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        offset_y = (28 - h) // 2 - rmin
        offset_x = (28 - w) // 2 - cmin
        centered = np.zeros((28, 28), dtype=np.uint8)
        sy1, sy2 = max(0, -offset_y), min(26, 26 - offset_y)
        sx1, sx2 = max(0, -offset_x), min(26, 26 - offset_x)
        dy1, dy2 = max(0, offset_y), min(28, 28 + offset_y)
        dx1, dx2 = max(0, offset_x), min(28, 28 + offset_x)
        h_copy = min(sy2 - sy1, dy2 - dy1)
        w_copy = min(sx2 - sx1, dx2 - dx1)
        if h_copy > 0 and w_copy > 0:
            centered[dy1:dy1+h_copy, dx1:dx1+w_copy] = arr[sy1:sy1+h_copy, sx1:sx1+w_copy]
    else:
        centered = np.zeros((28, 28), dtype=np.uint8)

    pil_img = PILImage.fromarray(centered)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(pil_img)

    arr[:1, :] = 0
    arr[-1:, :] = 0
    arr[:, :1] = 0
    arr[:, -1:] = 0

    return arr

class Segment:
    def __init__(self, segment_id, length, slope, direction, line_id):
        self.segment_id = segment_id
        self.length = length
        self.slope = slope
        self.direction = direction
        self.line_id = line_id


class Line:
    def __init__(self, line_id, image_id):
        self.line_id = line_id
        self.image_id = image_id
        self.segments = []

    def add_segment(self, segment):
        pass


class Image:
    def __init__(self, num_lines, thickness, image_id):
        self.num_lines = num_lines
        self.thickness = thickness
        self.image_id = image_id
        self.lines = [Line(line_id, image_id) for line_id in range(1, num_lines+1)]
        self.grid = np.zeros((26,26), dtype=np.int32)
        #print("num_lines ", num_lines)
        #print("thickness ", thickness)

        # chose startpos for line
        for line_id, line in enumerate(self.lines):
            #print("adding line ", line_id)
            self.add_line(line_id)


    def add_line(self, line_id):
        temp_grid = np.zeros(self.grid.shape).astype(self.grid.dtype)
        oldgrid = np.copy(self.grid)
        for i in range(10):
            buffer = self.thickness - 1
            y = random.randint(buffer, max(buffer+1, 26 - buffer - 1))
            x = random.randint(buffer, max(buffer+1, 26 - buffer - 1))
            #print("startpos: (", y, x, ")")

            if self.grid[y, x] == 0:
                # calculate max lenth of line
                max_length = max(1, (13 * 13) // (self.thickness * self.thickness))
                num_segments = random.randint(1, max_length + 1)

                ny = y
                nx = x

                for segment_id in range(num_segments):
                    if max_length < 1:
                        break

                    length = random.randint(1, max(max(y, 26-y), max(x, 26-x)))
                    slope = random.randint(1, length//self.thickness + 1) / length
                    direction = random.randint(0, 7)
                    thickness = self.thickness

                    # recalculate length
                    newlength = 0
                    ty = y
                    tx = x
                    while True:
                        if ty < 0 or ty > 26:
                            break
                        if tx < 0 or tx > 26:
                            break
                        newlength += thickness//2+1

                        if direction == 0:
                            ty = ty + slope
                            tx = tx + thickness//2+1
                        elif direction == 1:
                            ty = ty - thickness//2+1
                            tx = tx + slope
                        elif direction == 2:
                            ty = ty - thickness//2+1
                            tx = tx - slope
                        elif direction == 3:
                            ty = ty - slope
                            tx = tx - thickness//2+1
                        elif direction == 4:
                            ty = ty + slope
                            tx = tx - thickness//2+1
                        elif direction == 5:
                            ty = ty + thickness//2+1
                            tx = tx - slope
                        elif direction == 6:
                            ty = ty + thickness//2+1
                            tx = tx + slope
                        elif direction == 7:
                            ty = ty - slope
                            tx = tx + thickness//2+1

                    length = newlength


                    #print("segment:", segment_id, " (length:", length, " slope:", slope, " direction:", direction, ")")

                    for step in range(length):
                        if direction == 0:
                            #print("step: ", step)
                            ny = ny + slope
                            nx = nx + thickness//2+1
                        elif direction == 1:
                            #print("step: ", step)
                            ny = ny - thickness//2+1
                            nx = nx + slope
                        elif direction == 2:
                            #print("step: ", step)
                            ny = ny - thickness//2+1
                            nx = nx - slope
                        elif direction == 3:
                            #print("step: ", step)
                            ny = ny - slope
                            nx = nx - thickness//2+1
                        elif direction == 4:
                            #print("step: ", step)
                            ny = ny + slope
                            nx = nx - thickness//2+1
                        elif direction == 5:
                            #print("step: ", step)
                            ny = ny + thickness//2+1
                            nx = nx - slope
                        elif direction == 6:
                            #print("step: ", step)
                            ny = ny + thickness//2+1
                            nx = nx + slope
                        elif direction == 7:
                            #print("step: ", step)
                            ny = ny - slope
                            nx = nx + thickness//2+1

                        dt = thickness//2
                        if slope < 1 and thickness < 1:
                            dt = 2

                        if np.any(temp_grid[int(ny):int(ny)+1, int(nx):int(nx)+1] == 1):
                            max_length = max_length - step
                            break

                        temp_grid[int(ny)-dt:int(ny)+dt+1, int(nx)-dt:int(nx)+dt+1] = 1

                    max_length = max_length - length


                    self.grid = np.clip(self.grid + temp_grid, 0, 1)

                break

            if i == 10:
                print("failed adding startpos for line ", line_id)

        #print(temp_grid)
        #print("\n")

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
        #num_lines = random.randint(
        thickness = random.randint(2, max(2, 13 // num_lines))
        image = Image(num_lines, thickness, 1)
        return postprocess(image.grid)


def main():
    world = World()
    writer = imageio.get_writer("output.mp4", fps=2, macro_block_size=1)
    for i in range(100):
        grid = world.get_random_image()
        writer.append_data(grid)
        print(f"Frame {i+1}/100")
    writer.close()
    print("Done. Video saved to output.mp4")


if __name__ == "__main__":
    main()
