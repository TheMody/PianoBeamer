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

def detect_keyboard_and_postprocess(image, visualize = True, refine = True):
    # detects a keyboard from an overhead view as best as possible
    mask = detect_keyboard(image) # this is an image filled with 1s and 0s, where 1s are the pixels that are part of the keyboard

    if np.sum(mask) == 0:
        print("No keyboard detected in the image.")
        return None
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
    for i in range(-45,45,1):
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
                if  num_pixels >= mask_pixels:
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





        

    if visualize:
        image = image.astype(np.float32)  # Convert image to float32 for overlaying mask
        image[:,:,1] = image[:,:,1] + mask * 255  # Overlay the mask on the red channel
        image = np.clip(image, 0, 255)  # Ensure pixel values are within valid range
        image = image.astype(np.uint8)  # Convert back to uint8 for display



    return (pt1, pt2, pt3, pt4)

def project_points(keyboard_contour, corners):
    """
    Projects the keyboard contour points into the beamer perspective based on the detected corners.
    Args:
        keyboard_contour (list): List of points representing the keyboard contour.
        corners (list): List of points representing the detected corners.
    Returns:
        tuple: Transformed points in the beamer perspective.
    """

    pt1, pt2, pt3, pt4 = keyboard_contour

    # cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, pt2, pt3, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, pt3, pt4, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, pt4, pt1, (0,0,255), 3, cv2.LINE_AA)

#   cv2.imwrite("output_image.png", image)

    cpt1, cpt2, cpt3, cpt4 = corners[0], corners[1], corners[2], corners[3] #(top-left, top-right, bottom-right, bottom-left)
    cpt1 = np.asarray(tuple(map(int, cpt1)))
    cpt2 = np.asarray(tuple(map(int, cpt2)))
    cpt3 = np.asarray(tuple(map(int, cpt3)))
    cpt4 = np.asarray(tuple(map(int, cpt4)))

    # cv2.line(image, cpt1, cpt2, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, cpt2, cpt3, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, cpt3, cpt4, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, cpt4, cpt1, (0,0,255), 3, cv2.LINE_AA)

    #first warp keyboard points into beamer perspective
    #calculate transfrom from camera coordinates to beamer coordinates
    
    #first shift coordinates to origin of cpt1
    pt1 = pt1 - cpt1
    pt2 = pt2 - cpt1
    pt3 = pt3 - cpt1
    pt4 = pt4 - cpt1

    #calculate the basis vectors for the beamer coordinates
    first_basis_vector = np.asarray(cpt2-cpt1) 
    first_basis_vector = first_basis_vector / np.linalg.norm(first_basis_vector) * (b_width / np.linalg.norm(first_basis_vector))
    second_basis_vector = np.asarray(cpt4-cpt1) 
    second_basis_vector = second_basis_vector /np.linalg.norm(second_basis_vector)  * (b_height/np.linalg.norm(second_basis_vector))

    newpt1 = (pt1 @ first_basis_vector , pt1 @ second_basis_vector)
    newpt2 = (pt2 @ first_basis_vector , pt2 @ second_basis_vector)
    newpt3 = (pt3 @ first_basis_vector , pt3 @ second_basis_vector)
    newpt4 = (pt4 @ first_basis_vector , pt4 @ second_basis_vector) 

    return newpt1, newpt2, newpt3, newpt4