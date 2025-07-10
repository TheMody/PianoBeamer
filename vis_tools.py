import numpy as np
from PIL import Image


def visualize_prediction(image, mask, pred_mask = None, path = "image.png"):
    """Visualizes the original image, ground truth mask, and predicted mask."""
    image = image + np.min(image)
    image = image / np.max(image) 

    mask = np.logical_not(mask)  # Invert mask to visualize as white on black
    image[:,:,0] = image[:, :, 0] + mask 
    if pred_mask is not None:
        pred_mask = np.logical_not(pred_mask)  # Invert predicted mask
        image[:,:,1] = image[:, :, 1] + pred_mask 
    image = image / np.max(image)  # Normalize to [0, 1] for visualization
    # Convert to uint8 for visualization
    image = (image * 255).astype(np.uint8)
    #save img with PIL
   # maskvis = np.zeros_like(image) 
   # maskvis[:,:,0] = mask*255

    img = Image.fromarray(image)
    img.save(path)