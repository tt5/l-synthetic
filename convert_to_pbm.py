import os
import glob
import numpy as np

src_dir = 'data'
dst_dir = 'data_pbm'
threshold = 128

for class_path in sorted(glob.glob(os.path.join(src_dir, 'class_*'))):
    if not os.path.isdir(class_path):
        continue
    class_name = os.path.basename(class_path)
    dst_class_dir = os.path.join(dst_dir, class_name)
    os.makedirs(dst_class_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(class_path, '*.npy')))
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {f}, got shape {arr.shape}")
        h, w = arr.shape
        # Threshold to binary: 0 (white) or 1 (black)
        bin_arr = (arr >= threshold).astype(np.uint8)
        # Pack into P4 format: each byte holds 8 pixels, MSB-first.
        flat = bin_arr.reshape(-1)
        pad = (8 - (flat.size % 8)) % 8
        if pad:
            flat = np.pad(flat, (0, pad), constant_values=0)
        packed = np.packbits(flat)
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(dst_class_dir, base + '.pbm')
        with open(out_path, 'wb') as out:
            out.write(b'P4\n')
            out.write((f'# {class_name}/{base}.npy\n').encode('ascii'))
            out.write(f'{w} {h}\n'.encode('ascii'))
            out.write(packed.tobytes())
print('Done.')
