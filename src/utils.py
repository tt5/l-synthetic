import numpy as np
from PIL import Image as PILImage, ImageFilter

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
        is_full_image = (
            rmin == 0 and rmax == IMG_SIZE - 1 and cmin == 0 and cmax == IMG_SIZE - 1
        )
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
            centered[dy1 : dy1 + h_copy, dx1 : dx1 + w_copy] = arr[
                sy1 : sy1 + h_copy, sx1 : sx1 + w_copy
            ]
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
