
import cv2
#from config import WEBCAM_ID

def capture_img(cam_index: int = 4, backend = cv2.CAP_ANY):
    # 1. Open the device in the constructor
    cap = cv2.VideoCapture(cam_index, backend)

    if not cap.isOpened():          # <- returns immediately if the open failed
        raise IOError(f"Cannot open camera (index {cam_index})")

    try:
        # 2. Grab one frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")
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

    # Show the image with contours
    cv2.namedWindow("Keyboard and Beamer Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("Keyboard and Beamer Contours", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()