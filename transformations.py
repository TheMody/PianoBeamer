import numpy as np
from model_training.identify_img import detect_keyboard
import cv2
from config import *
def triangle_area(p1, p2, p3):
    """
    Return the area of a triangle given three 2-D points.

    Parameters
    ----------
    p1, p2, p3 : tuple[float, float]
        The vertices, e.g. (x, y).

    Returns
    -------
    float
        Triangle area (always non-negative).
    """
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return abs(
        (x1 * (y2 - y3) +
         x2 * (y3 - y1) +
         x3 * (y1 - y2)) * 0.5
    )

def calculate_area(pt1, pt2, pt3, pt4):
    """
    Calculate the area of a quadrilateral given its four corner points.
    Uses the shoelace formula to calculate the area.
    """
    area = triangle_area(pt1, pt2, pt3) + triangle_area(pt1, pt3, pt4)
    return area

def extract_cornerpoints_from_mask(mask, refine = True):

    #find the four edge points of the keyboard
    #first get all indices of the mask where the value is 1 and try out different rotations
    def find_keyboard_axis_aligned(mask):
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

        #calculate area of the keyboard
        area = (max_x - min_x) * (max_y - min_y)

        return pt1, pt2, pt3, pt4, area
    
    min_area = 1e9
    best_rot = 0
    #try different rotations
    for i in range(-22,22,1):
        #define rotation matrix 2D
        angle = i 
        rotation_matrix = cv2.getRotationMatrix2D((mask.shape[1] // 2, mask.shape[0] // 2), angle, 1)
        #rotate the mask
        rotated_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, rotation_matrix, (mask.shape[1], mask.shape[0]))
        rotation_matrix_inv = cv2.getRotationMatrix2D((mask.shape[1] // 2, mask.shape[0] // 2), -angle, 1)

        pt1, pt2, pt3, pt4, area = find_keyboard_axis_aligned(rotated_mask // 255)
        if area < min_area:
            min_area = area
            pts_rot = np.array([pt1,pt2,pt3,pt4], dtype=np.float32).reshape(-1, 1, 2)
            pts_orig = cv2.transform(pts_rot, rotation_matrix_inv)  # still (N,1,2)
            pts_orig = pts_orig.reshape(-1, 2)        # drop the dummy dimension
            best_rot = i
            best_pt1, best_pt2, best_pt3, best_pt4 = pts_orig.astype(int)



    if refine:
        rotation_matrix = cv2.getRotationMatrix2D((mask.shape[1] // 2, mask.shape[0] // 2), best_rot, 1)
        rotated_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, rotation_matrix, (mask.shape[1], mask.shape[0]))
        pt1, pt2, pt3, pt4, area = find_keyboard_axis_aligned(rotated_mask // 255)
        rotation_matrix_inv = cv2.getRotationMatrix2D((mask.shape[1] // 2, mask.shape[0] // 2), -best_rot, 1)
        #try out slightly smaller area
        pts_available = [0,1,2,3,4,5,6,7]
        pts = [pt1, pt2, pt3, pt4]
        step = 2
        mask_pixels = np.sum(rotated_mask == 255)
        while len(pts_available) >= 1:
            for pts_idx in pts_available:
                try_out_pts = pts.copy()
                if pts_idx // 4 == 0:
                    first, second = 1, 0
                else:
                    first, second = 0, 1
                if pts_idx % 4 == 0:
                    try_out_pts[pts_idx % 4] = (try_out_pts[pts_idx % 4][0] + step * first, try_out_pts[pts_idx % 4][1] + step * second)
                if pts_idx % 4 == 1:
                    try_out_pts[pts_idx % 4] = (try_out_pts[pts_idx % 4][0] - step* first, try_out_pts[pts_idx % 4][1] + step* second)
                if pts_idx % 4 == 2:
                    try_out_pts[pts_idx % 4] = (try_out_pts[pts_idx % 4][0] - step* first, try_out_pts[pts_idx % 4][1] - step* second)
                if pts_idx % 4 == 3:
                    try_out_pts[pts_idx % 4] = (try_out_pts[pts_idx % 4][0] + step* first, try_out_pts[pts_idx % 4][1] - step* second)
                
               # area = calculate_area(*try_out_pts)
                #also check if number of pixels in the mask is still the same
                #first select all pixels in the mask that are in the area marked by the tryoutpts
                mask_copy = np.zeros_like(rotated_mask, dtype=np.uint8)
                cv2.fillConvexPoly(mask_copy, np.array(try_out_pts, dtype=np.int32), 2)

                num_pixels = np.sum((mask_copy + rotated_mask) == 1)
                if  num_pixels >= mask_pixels - 3:  # allow for a small margin of error
                  #  min_area = area
                    pts = try_out_pts
                   # print(f"new best pts {pts_idx} num_pixels {num_pixels} mask_pixels {mask_pixels}")
                 #   print(pts)
                else:
                   # print(f"Skipping pts {pts_idx}num_pixels {num_pixels} mask_pixels {mask_pixels}")
                    pts_available.remove(pts_idx)

        pts_rot = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        pts_orig = cv2.transform(pts_rot, rotation_matrix_inv)  # still (N,1,2)
        pts_orig = pts_orig.reshape(-1, 2)        # drop the dummy dimension
        best_pt1, best_pt2, best_pt3, best_pt4 = pts_orig.astype(int)
        #  best_pt1, best_pt2, best_pt3, best_pt4 = pts
        
    pt1, pt2, pt3, pt4 = best_pt1, best_pt2, best_pt3, best_pt4

    return pt1, pt2, pt3, pt4

def detect_beamer_area(image, background_img, threshold = marker_threshold):
    '''the image should be a camera picture where the beamer projected just a white screen, while on the background_img the beamer projected a black screen. The difference should be easy to calculate.

    image: the image captured by the camera
    background_img: the image captured by the camera when the beamer projected a black screen
    '''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(float)
    bg_gray = bg_gray.astype(float)
   # gray = np.clip(gray - bg_gray, 0, 255)

    #calculate proportional increase
    gray = gray / bg_gray
    gray[bg_gray == 0] = 1.0  # reset pixels where bg_gray is 0 to avoid division by zero
    # cv2.imshow("Beamer Area Detection", gray.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #blur the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #threshold the image to create a mask
    mask = gray > threshold  # Adjust threshold as needed

    pt1, pt2, pt3, pt4 = extract_cornerpoints_from_mask(mask, refine=True)   

    # display_img = mask.astype(np.uint8) * 255
    # display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # cv2.line(display_img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(display_img, pt2, pt3, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(display_img, pt3, pt4, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(display_img, pt4, pt1, (0,0,255), 3, cv2.LINE_AA)

    # cv2.imshow("Beamer Area Detection", display_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return (pt1, pt2, pt3, pt4)

def detect_keyboard_and_postprocess(image, refine = True):
    # detects a keyboard from an overhead view as best as possible
    mask = detect_keyboard(image) # this is an image filled with 1s and 0s, where 1s are the pixels that are part of the keyboard

    if np.sum(mask) == 0:
        print("No keyboard detected in the image.")
        return None
    
    # Extract corner points from the mask
    pt1, pt2, pt3, pt4 = extract_cornerpoints_from_mask(mask, refine=refine)
    
    # if visualize:
    #     image = image.astype(np.float32)  # Convert image to float32 for overlaying mask
    #     image[:,:,1] = image[:,:,1] + mask * 255  # Overlay the mask on the red channel
    #     image = np.clip(image, 0, 255)  # Ensure pixel values are within valid range
    #     image = image.astype(np.uint8)  # Convert back to uint8 for display

    return (pt1, pt2, pt3, pt4)

