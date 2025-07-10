#!/usr/bin/env python3
"""
make_mask.py  –  convert a LabelMe JSON polygon to a 0/1 segmentation mask

Usage:
    python make_mask.py annotation.json mask.png
"""
import os
import json
import sys
from pathlib import Path

import cv2          # pip install opencv-python
import numpy as np
from PIL import Image  # pip install pillow


def polygon_to_mask(shape, img_h, img_w):
    """
    Returns a binary mask (numpy array, uint8) for a single polygon shape.
    """
    # points arrive as [[x1, y1], [x2, y2], ...]; cv2 expects int32, (N,1,2)
    pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)  # fill with 1s inside the polygon
    return mask


def main(json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    img_h = ann["imageHeight"]
    img_w = ann["imageWidth"]

    # Start with an all-zero background
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Iterate over shapes; only the label “keyboard” gets value 1
    for shape in ann.get("shapes", []):
        if shape.get("label") == "keyboard" and shape.get("shape_type") == "polygon":
            full_mask |= polygon_to_mask(shape, img_h, img_w)  # logical OR in case of overlap

    # Save as a PNG (lossless, keeps values 0/1)
    Image.fromarray(full_mask * 255).save(out_path)
    print(f"Saved mask to {out_path}")


if __name__ == "__main__":
    path = "data/"
    for file in os.listdir(path + "jsons"):
        if file.endswith('.json'):
            json_file = os.path.join(path + "jsons", file)
            mask_file = os.path.join(path + "masks", file.replace('.json', '.png'))
            Path(mask_file).parent.mkdir(parents=True, exist_ok=True)
            main(json_file, mask_file)