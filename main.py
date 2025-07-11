import cv2
import numpy as np
from identify_img import detect_keyboard
import math
from marker import detect_four_markers
from keyboard_vis_cv import PianoKeyboardCV
from config import *

def detect_keyboard_and_postprocess(image):
    # detects a keyboard from an overhead view as best as possible
    # first tries to find the keyboard by finding the largest and brightest almost white area
    mask = detect_keyboard(image) # this is an image filled with 1s and 0s, where 1s are the pixels that are part of the keyboard
   # print(mask)
    #do cv line detection on the mask image:

    #find the four edge points of the keyboard
    #first get all indices of the mask where the value is 1

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
    #try different rotations
    for i in range(45):
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
            best_pt1, best_pt2, best_pt3, best_pt4 = pts_orig.astype(int)

    pt1, pt2, pt3, pt4 = best_pt1, best_pt2, best_pt3, best_pt4

    image = image.astype(np.float32)  # Convert image to float32 for overlaying mask
    image[:,:,1] = image[:,:,1] + mask * 255  # Overlay the mask on the red channel
    image = np.clip(image, 0, 255)  # Ensure pixel values are within valid range
    image = image.astype(np.uint8)  # Convert back to uint8 for display
    # cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, pt2, pt3, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, pt3, pt4, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(image, pt4, pt1, (0,0,255), 3, cv2.LINE_AA)
    

    return (pt1, pt2, pt3, pt4)

if __name__ == "__main__":
    # Load an image
    image = cv2.imread('images/example.jpg')
    marker_img = cv2.imread('images/four_markers.png')
    # rescale image to a reasonable size
    if image is None:
        print("Error: Could not load image.")
    else:
        height, width = image.shape[:2]
        scale_factor = 1200 / max(height, width)
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
        # Detect keyboard
        keyboard_contour = detect_keyboard_and_postprocess(image)

        #add marker image to example
        marker_img = cv2.resize(marker_img, (int(image.shape[1]*0.9), int(image.shape[0]*0.9)))
        # Overlay the marker image on the original image
        combined_image = image.copy()
        combined_image = combined_image.astype(np.float32)  # Convert to float32 for blending
        offset = (50,50)
        combined_image[offset[0]:offset[0]+marker_img.shape[0], offset[1]:offset[1]+marker_img.shape[1]] = combined_image[offset[0]:offset[0]+marker_img.shape[0], offset[1]:offset[1]+marker_img.shape[1]]*0.5 + marker_img*0.5

        combined_image = np.clip(combined_image, 0, 255)  # Ensure pixel values are within valid range
        #display image

        corners = detect_four_markers(combined_image)#,background_img=image*0.9)

        pt1, pt2, pt3, pt4 = keyboard_contour

        cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(image, pt2, pt3, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(image, pt3, pt4, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(image, pt4, pt1, (0,0,255), 3, cv2.LINE_AA)

     #   cv2.imwrite("output_image.png", image)

        cpt1, cpt2, cpt3, cpt4 = corners[0], corners[1], corners[2], corners[3] #(top-left, top-right, bottom-right, bottom-left)
        cpt1 = np.asarray(tuple(map(int, cpt1)))
        cpt2 = np.asarray(tuple(map(int, cpt2)))
        cpt3 = np.asarray(tuple(map(int, cpt3)))
        cpt4 = np.asarray(tuple(map(int, cpt4)))

        cv2.line(image, cpt1, cpt2, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(image, cpt2, cpt3, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(image, cpt3, cpt4, (0,0,255), 3, cv2.LINE_AA)
        cv2.line(image, cpt4, cpt1, (0,0,255), 3, cv2.LINE_AA)

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

        #these are the coordinates for the beamer to display the keyboard

        #get piano img
        kb = PianoKeyboardCV().img
        src_quad = np.array([[0,0], [kb.shape[1], 0], [kb.shape[1], kb.shape[0]], [0, kb.shape[0]]], dtype=np.float32)
        dst_quad = np.array([newpt1, newpt2, newpt3, newpt4], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_quad, dst_quad)
        h, w = kb.shape[:2]
        kb_new = np.zeros((b_height, b_width, 3), dtype=np.uint8)  # Create a blank image for the keyboard
        kb_new[:h, :w] = kb
        display_img = cv2.warpPerspective(kb_new, H, (b_width, b_height))

        cv2.imshow("Source", display_img )  # Show the original image with the mask overlayed
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # mask = np.zeros((h, w), dtype=np.uint8)
        # cv2.fillConvexPoly(mask, dst_quad.astype(int), 255)

        # composited = frame.copy()
        # composited[mask == 255] = warped[mask == 255]
        # print(kb.shape)

      #  print("New points:", newpt1, newpt2, newpt3, newpt4)

        cv2.imshow("Source", image )  # Show the original image with the mask overlayed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
      #  print("Detected corners:", corners)


    
    