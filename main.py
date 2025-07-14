import cv2
import numpy as np

import math
from marker import detect_four_markers
from keyboard_vis_cv import PianoKeyboardCV
from config import *
from transformations import detect_keyboard_and_postprocess, project_points
import mido
from keyboard_vis_cv import animate, extract_events

if __name__ == "__main__":
    # Load an image
    image = cv2.imread('images/challenging_example.png')
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


        # Create a dummy input we can use as a test
        marker_img = cv2.resize(marker_img, (int(image.shape[1]*0.9), int(image.shape[0]*0.9)))
        combined_image = image.copy()
        combined_image = combined_image.astype(np.float32)  # Convert to float32 for blending
        offset = (50,50)
        combined_image[offset[0]:offset[0]+marker_img.shape[0], offset[1]:offset[1]+marker_img.shape[1]] = combined_image[offset[0]:offset[0]+marker_img.shape[0], offset[1]:offset[1]+marker_img.shape[1]]*0.5 + marker_img*0.5
        combined_image = np.clip(combined_image, 0, 255)  # Ensure pixel values are within valid range

        #detect the four markers in the image
        corners = detect_four_markers(combined_image)#,background_img=image*0.9)

        #these are the corner coordinates for the beamer to display the keyboard
        newpt1, newpt2, newpt3, newpt4 = project_points(keyboard_contour, corners)
        
        #calculate the transformation for the keyboard to display 
        kb = PianoKeyboardCV(start_midi=21, num_keys=num_key)
        src_quad = np.array([[0,0], [kb.width, 0], [kb.width, kb.height], [0, kb.height]], dtype=np.float32)
        dst_quad = np.array([newpt1, newpt2, newpt3, newpt4], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_quad, dst_quad)
        kb.update_transform(H)

        #animate the keyboard with a midi file
        midi = mido.MidiFile("midi_files/smoke.mid")
        events = extract_events(midi)
        animate(events, kb, lookahead=2.5)

    


    
    