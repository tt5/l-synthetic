#!/usr/bin/env python3

import random
import numpy as np
from skimage.measure import label
from scipy import ndimage

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
        self.grid = np.zeros((27,27), dtype=np.int32)
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
            y = random.randint(buffer, max(buffer+1, 27 - buffer - 1))
            x = random.randint(buffer, max(buffer+1, 27 - buffer - 1))
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
        num_lines = random.randint(0, 13) + 1
        thickness = random.randint(0, 13 // num_lines) + 1
        image = Image(num_lines, thickness, 1)
        return image.grid


def main():
    world = World()
    new_random_image = world.get_random_image()
    #print("\n---\n")
    print(new_random_image)


if __name__ == "__main__":
    main()
