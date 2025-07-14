
import cv2

def capture_img():
    # 1. Open a connection to the first camera (index 0).
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # on Linux/macOS you can omit cv2.CAP_DSHOW

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
        #raise IOError("Cannot open camera")

    try:
        # 2. Read one frame.
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")
        # cap.release()
        # cv2.destroyAllWindows()
        # 3. Process or save the frame as needed.
        return frame      # example: save to disk
        # cv2.imshow("Current frame", frame)    # or display in a window
        # cv2.waitKey(0)

    finally:
        # 4. Release the camera and any OpenCV windows.
        cap.release()
