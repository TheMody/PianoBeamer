
import cv2
from config import *
#from config import WEBCAM_ID

def list_available_cams(max_index: int = 10, backend=cv2.CAP_ANY) -> list[int]:
    """
    Return a list of camera indexes that can be opened with OpenCV.

    Parameters
    ----------
    max_index : int
        Highest index (exclusive) to probe, e.g. 10 means 0‥9.
    backend : int
        OpenCV backend flag (default: cv2.CAP_ANY).

    Notes
    -----
    • Adjust `max_index` if you expect more than 10 devices.  
    • On Windows you might prefer `backend=cv2.CAP_DSHOW` or `cv2.CAP_MSMF`
      to speed up probing.
    """
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found

def capture_img(cam_index: int = 4, backend = cv2.CAP_ANY): #cv2.CAP_ANY):
    # 1. Open the device in the constructor
    cap = cv2.VideoCapture(cam_index, backend)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)

    if not cap.isOpened():          # <- returns immediately if the open failed
        print(f"Cannot open camera (index {cam_index})")
        available_list = list_available_cams()  # <- probe available cameras
        print("Available camera indexes:")
        print(available_list)
        return None

    try:
        # 2. Grab one frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")
        
        print("captured image of shape",frame.shape)
        return frame
    finally:
        cap.release()

def visualize_keyboard_and_beamer(image,keyboard_contour,beamer_contour):
    display_img = image.copy()

    pt1, pt2, pt3, pt4 = keyboard_contour

    # Draw the keyboard contour in green
    cv2.line(display_img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.line(display_img, pt2, pt3, (0,0,255), 3, cv2.LINE_AA)
    cv2.line(display_img, pt3, pt4, (0,0,255), 3, cv2.LINE_AA)
    cv2.line(display_img, pt4, pt1, (0,0,255), 3, cv2.LINE_AA)

    # Draw the beamer contour in blue
    pt1, pt2, pt3, pt4 = beamer_contour

    cv2.line(display_img, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
    cv2.line(display_img, pt2, pt3, (0,255,0), 3, cv2.LINE_AA)
    cv2.line(display_img, pt3, pt4, (0,255,0), 3, cv2.LINE_AA)
    cv2.line(display_img, pt4, pt1, (0,255,0), 3, cv2.LINE_AA)

    #save the image with contours   
    cv2.imwrite("keyboard_beamer_contours.png", display_img)

    return display_img
    # # Show the image with contours
    # cv2.namedWindow("Keyboard and Beamer Contours", cv2.WINDOW_NORMAL)
    # cv2.imshow("Keyboard and Beamer Contours", display_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def save_parameters(keyboard_contour, beamer_contour,camera_distortion = None, filename="calibration_params.txt"):
    """
    Save the keyboard and beamer contours to a text file.
    """
    with open(filename, "w") as f:
        f.write("Keyboard Contour:\n")
        for pt in keyboard_contour:
            f.write(f"{pt[0]},{pt[1]}\n")
        
        f.write("\nBeamer Contour:\n")
        for pt in beamer_contour:
            f.write(f"{pt[0]},{pt[1]}\n")

        if camera_distortion is not None:
            f.write("\nCamera Distortion:\n")
            for param in camera_distortion:
                f.write(f"{param}\n")

def load_parameters(filename="calibration_params.txt"):
    """
    Load the keyboard and beamer contours from a text file.
    """
    keyboard_contour = []
    beamer_contour = []
    camera_distortion = []

    with open(filename, "r") as f:
        lines = f.readlines()
        section = None
        for line in lines:
            line = line.strip()
            if line == "Keyboard Contour:":
                section = "keyboard"
            elif line == "Beamer Contour:":
                section = "beamer"
            elif line == "Camera Distortion:":
                section = "distortion"
            elif section == "keyboard":
                if line:
                    x, y = map(int, line.split(","))
                    keyboard_contour.append((x, y))
            elif section == "beamer":
                if line:
                    x, y = map(int, line.split(","))
                    beamer_contour.append((x, y))
            elif section == "distortion":
                if line:
                    camera_distortion.append(float(line))

    return keyboard_contour, beamer_contour, camera_distortion