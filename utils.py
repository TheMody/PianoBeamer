
import cv2
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

def capture_img(cam_index: int = 4, backend = cv2.CAP_DSHOW): #cv2.CAP_ANY):
    # 1. Open the device in the constructor
    cap = cv2.VideoCapture(cam_index, backend)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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