#!/usr/bin/env python
"""
Minimal inference script for a fine-tuned Mask2Former model.

This script loads a model checkpoint and its corresponding processor configuration
to perform semantic segmentation on a single image.

Usage:

$ python inference.py \
    --image_path /path/to/your/image.jpg \
    --model_path ./checkpoints/best_model.pth \
    --output_path ./output_visualization.png \
    --num_classes 21 \
    --model_name "facebook/mask2former-swin-base-IN21k-ade-semantic"
"""
import json
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor
from model_training.model import custom_model
from vis_tools import visualize_prediction
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

MAXSIZE = 384

def detect_keyboard(image):
    image = np.array(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the checkpoint
    checkpoint = torch.load("model_training/checkpoints/best_model.pth", map_location=device)

    # 2. Recreate the processor from the saved configuration
    processor_config = json.loads(checkpoint["processor"])
    processor = Mask2FormerImageProcessor.from_dict(processor_config)

    # 3. Recreate the model architecture and load the fine-tuned weights
    model = custom_model(processor=processor)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Load and preprocess the input image
    # print(f"Loading and preprocessing image: {img_pth}")
    # image = Image.open(img_pth).convert("RGB")
   # print(image.shape)
    original_size = image.shape[0:-1]  # (height, width)
   # print(f"Original image size: {original_size}")



    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
   # print(pixel_values.shape)

    # 5. Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # 6. Post-process the output to get the final segmentation mask
    # Upsample logits to the original image size for better quality
    upscaled_logits = torch.nn.functional.interpolate(
        outputs,
        size=original_size, # (height, width)
        mode='bilinear',
        align_corners=False
    )
    # Get the predicted class for each pixel
    pred_mask = upscaled_logits.argmax(dim=1)

    #resize mask to same size as input image
    pred_mask = torch.nn.functional.interpolate(
        pred_mask.unsqueeze(1).float(),
        size=original_size,
        mode='nearest'
    ).squeeze(1)

    #invert mask
    pred_mask = (pred_mask == 0).squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

    return pred_mask

if __name__ == "__main__":
    image = Image.open("example.jpg").convert("RGB")
    if image.size[0] > MAXSIZE:
        image = image.resize((MAXSIZE, int(image.size[1]*(MAXSIZE/image.size[1])) ), Image.BILINEAR)
    if image.size[1] > MAXSIZE:
        image = image.resize(( int(image.size[0]*(MAXSIZE/image.size[0])),MAXSIZE ), Image.BILINEAR)
    pred_mask = detect_keyboard(image)
    visualize_prediction(np.array(image), pred_mask, path = "output_visualization.png")