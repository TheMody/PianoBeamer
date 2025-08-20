from time import sleep

from regex import T
from utils import capture_img
import cv2 as cv
import numpy as np
from config import WEBCAM_ID

def undistort_camera():
    img_list  = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []  # 2d points in image plane.
    objpoints = []  # 3d points in real world space.

    tries = 0
    while len(imgpoints) < 12 and tries < 100:
        tries += 1
        sleep(0.5)
        img = capture_img(cam_index=WEBCAM_ID, undistort=False)
        if img is not None:
            img_list.append(img)
        else:
            print("Failed to capture image, retrying...")
            continue
        
        #display the captured image
        cv.imshow("Captured Image", img)
        cv.waitKey(1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        else:
            print("Chessboard corners not found, retrying...")

    #calibrate the camera
    print("Calibrating camera with collected images...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #saving the calibration parameters
    np.savez("camera_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Camera calibration parameters saved to camera_calibration.npz")