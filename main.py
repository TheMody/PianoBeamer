import cv2
import numpy as np

from marker import detect_four_markers
from keyboard_vis_cv import PianoKeyboardCV
from config import *
from transformations import detect_keyboard_and_postprocess, detect_beamer_area
import mido
from keyboard_vis_cv import animate, extract_events
from utils import capture_img, visualize_keyboard_and_beamer
import time


def recalibrate(kb,keyboard_edge_points, beamer_edge_points):
        src_quad = np.array([[0,0], [kb.width, 0], [kb.width, kb.height], [0, kb.height]], dtype=np.float32)
        dst_quad = np.array(keyboard_edge_points, dtype=np.float32)
        H_2 = cv2.getPerspectiveTransform(src_quad, dst_quad)

        #calculate the transformation from camera space to beamer space
        src_quad = np.array(beamer_edge_points, dtype=np.float32)
        dst_quad = np.array([[0,0], [b_width, 0], [b_width, b_height], [0, b_height]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_quad, dst_quad)
        kb.update_transform(H, H_2)
        print("calibrated successfully")

        return

def setup_and_calibrate(test = test):
    if test:
        image = cv2.imread('images/challenging_example.png')
        image = cv2.resize(image, (c_width, c_height))
        print("loaded images")
    else:
        black_img = np.zeros((b_height, b_width, 3), dtype=np.uint8)
        cv2.namedWindow("black_img", cv2.WINDOW_NORMAL)          # create once, before first imshow
        cv2.setWindowProperty("black_img",
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN)
        cv2.imshow("black_img",black_img)
        cv2.waitKey(250)
        image = capture_img(WEBCAM_ID)
        cv2.waitKey(50) 
        cv2.destroyAllWindows()
        print("captured image")

    # rescale image to a reasonable size
    if image is None:
        print("Error: Could not load/capture image.")
    else:
        if test:
            # Create a dummy input we can use as a test
            if marker_Mode == "marker":
                marker_img = cv2.imread('images/four_markers.png')
                marker_img = cv2.resize(marker_img, (int(image.shape[1]*0.9), int(image.shape[0]*0.9)))
            else:
                marker_img = np.ones((int(image.shape[0]*0.5), int(image.shape[1]*0.5), 3), dtype=np.uint8) * 255  # Create a white image for testing
            combined_image = image.copy()
            combined_image = combined_image.astype(np.float32)  # Convert to float32 for blending
            offset = (250,250)
            combined_image[offset[0]:offset[0]+marker_img.shape[0], offset[1]:offset[1]+marker_img.shape[1]] = combined_image[offset[0]:offset[0]+marker_img.shape[0], offset[1]:offset[1]+marker_img.shape[1]]*0.5 + marker_img*0.5
            combined_image = np.clip(combined_image, 0, 255)  # Ensure pixel values are within valid range
            combined_image = combined_image.astype(np.uint8)
        else:
            # Show a marker image for calibration
            if marker_Mode == "marker":
                marker_img = cv2.imread('images/four_markers.png')
                marker_img = cv2.resize(marker_img,( b_width, b_height))
            else:
                marker_img = np.ones((b_height,b_width, 3), dtype=np.uint8) * 255 
            cv2.namedWindow("Marker_img", cv2.WINDOW_NORMAL)          
            cv2.setWindowProperty("Marker_img",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Marker_img",marker_img)
            cv2.waitKey(350)
            combined_image = capture_img(WEBCAM_ID)
            cv2.waitKey(50)
            cv2.destroyAllWindows()

        #detect the beamer projection area
        if marker_Mode == "marker":
            corners = detect_four_markers(combined_image)#,background_img=image*0.9)
        else:
            corners = detect_beamer_area(combined_image, image)

        #check if beamer corners make sense
        for point in corners:
            if point[0] > image.shape[1] or point[0] < 0:
                point[0] = max(0, min(point[0], image.shape[1]-1))
            if point[1] > image.shape[0] or point[1] < 0:
                point[1] = max(0, min(point[1], image.shape[0]-1))
        
        print("detected beamer view, if this is not correct, please adjust the beamer bounding box manually")

        #these are the corner coordinates for the beamer to display the keyboard
        keyboard_contour = detect_keyboard_and_postprocess(image)
        if keyboard_contour is None:
            print("Error: Could not detect keyboard in the image. Please adjust the keyboard bounding box manually.")
            keyboard_contour = np.array([[0,0], [b_width/2, 0], [b_width/2, b_height/2], [0, b_height/2]], dtype=np.int32)
        else:
            print("detected keyboard contour")

        #only used for debugging atm
        display_img = visualize_keyboard_and_beamer(combined_image, keyboard_contour, corners)

        #calculate the transformation for the keyboard to camera space 
        kb = PianoKeyboardCV(start_midi=21, num_keys=NUM_KEYS)
        src_quad = np.array([[0,0], [kb.width, 0], [kb.width, kb.height], [0, kb.height]], dtype=np.float32)
        dst_quad = np.array(keyboard_contour, dtype=np.float32)
        H_2 = cv2.getPerspectiveTransform(src_quad, dst_quad)

        #calculate the transformation from camera space to beamer space
        src_quad = np.array(corners, dtype=np.float32)
        dst_quad = np.array([[0,0], [b_width, 0], [b_width, b_height], [0, b_height]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_quad, dst_quad)
        kb.update_transform(H, H_2)
        print("calibrated successfully")

        return kb, combined_image, keyboard_contour, corners

def play_song(midi_file, kb, playback_speed=1.0):
    midi = mido.MidiFile(midi_file)
    events = extract_events(midi)
    print("extracted events from midi file", midi_file)
    animate(events, kb,playback_speed = playback_speed, lookahead=LOOKAHEAD)
    print("song finished")


if __name__ == "__main__":
    kb, _, _ , _ = setup_and_calibrate()
 
    play_song("midi_files/Fur Elise.mid", kb)
  #  play_song("midi_files/davy.mid", kb)
      

    


    
    