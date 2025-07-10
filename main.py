import cv2
import numpy as np
from identify_img import detect_keyboard
import math


def detect_keyboard_and_postprocess(image):
    # detects a keyboard from an overhead view as best as possible
    # first tries to find the keyboard by finding the largest and brightest almost white area
    mask = detect_keyboard(image) # this is an image filled with 1s and 0s, where 1s are the pixels that are part of the keyboard
   # print(mask)
    #do cv line detection on the mask image:

    #find the four edge points of the keyboard
    #first get all indices of the mask where the value is 1
    indices = np.argwhere(mask == 1)
    #now find the min and max indices in the x and y direction
    min_x = np.min(indices[:, 1])
    max_x = np.max(indices[:, 1])
    min_y = np.min(indices[:, 0])
    max_y = np.max(indices[:, 0])
    #now we have the four edge points of the keyboard
    pt1 = (min_x, min_y)  # top-left corner
    pt2 = (max_x, min_y)  # top-right corner
    pt3 = (max_x, max_y)  # bottom-right corner
    pt4 = (min_x, max_y)  # bottom-left corner

    image = image.astype(np.float32)  # Convert image to float32 for overlaying mask
    image[:,:,1] = image[:,:,1] + mask * 255  # Overlay the mask on the red channel
    image = np.clip(image, 0, 255)  # Ensure pixel values are within valid range
    image = image.astype(np.uint8)  # Convert back to uint8 for display
    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.line(image, pt2, pt3, (0,0,255), 3, cv2.LINE_AA)
    cv2.line(image, pt3, pt4, (0,0,255), 3, cv2.LINE_AA)
    cv2.line(image, pt4, pt1, (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Source", image )  # Show the original image with the mask overlayed
    # Draw the largest contour on the original image
    # cv2.putText(image, "Keyboard Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # # Show the result
    # cv2.imshow('Keyboard Detection', image.astype(np.uint8))
   # print(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 

if __name__ == "__main__":
    # Load an image
    image = cv2.imread('example.jpg')
    # rescale image to a reasonable size
    if image is None:
        print("Error: Could not load image.")
    else:
        height, width = image.shape[:2]
        scale_factor = 800 / max(height, width)
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
        # Detect keyboard
        keyboard_contour = detect_keyboard_and_postprocess(image)

    
    