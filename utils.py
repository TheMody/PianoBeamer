
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