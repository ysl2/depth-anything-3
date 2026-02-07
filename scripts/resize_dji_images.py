#!/usr/bin/env python3
"""
Resize DJI images to 512px width for DA3-Streaming processing.
"""
import os
import glob
from PIL import Image

src_dir = "/home/songliyu/Templates/DJI-Mini3-Pro/20260204/102MEDIA/fps_2/images"
dst_dir = "/home/songliyu/Documents/Depth-Anything-3/data/dji_20260204/images"
target_width = 512

os.makedirs(dst_dir, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.JPG")))

print(f"Found {len(img_files)} images")

for i, img_path in enumerate(img_files):
    img = Image.open(img_path)
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    img_resized = img.resize((target_width, new_h), Image.LANCZOS)

    basename = os.path.basename(img_path)
    new_name = os.path.splitext(basename)[0].lower() + ".jpg"
    img_resized.save(os.path.join(dst_dir, new_name))

    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(img_files)} images")

print(f"Done! Resized to {target_width}px width")
