import os
import cv2 as cv
from cv2 import aruco, imwrite
import numpy as np
from calibration import CameraCalibration

CALIBRATION_FILE_DEFAULT = "share/calibration/nexigo_webcam/cfg.xml"
MARKER_LENGTH = 40 # cm
MARKER_SEPARATION = 8 # cm 

# Create an aruco board
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
board = aruco.GridBoard_create(5, 7, MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)
aruco_params = aruco.DetectorParameters_create()

# Load our calibration
cam_calibration = CameraCalibration(CALIBRATION_FILE_DEFAULT)
cam_calibration.load_calibration()

vid = cv.VideoCapture(0)

while(True):
    # Capture video and undistort it
    ret,frame = vid.read()
    frame_undistorted = cam_calibration.undistort(frame)

    # Switch to grayscale
    frame_undistorted_gray = cv.cvtColor(frame_undistorted, cv.COLOR_BGR2GRAY)

    # detect aruco markers
    corners,ids,rejected = aruco.detectMarkers(frame_undistorted_gray, aruco_dict, parameters=aruco_params)
    aruco.refineDetectedMarkers(frame_undistorted_gray, board, corners, ids, rejected)

    im_with_aruco_board = frame_undistorted

    # if we found markers draw them and see whats up with the pose board
    if ids is not None:
        im_with_aruco_board = aruco.drawDetectedMarkers(frame_undistorted, corners, ids, (0,255,0))

    cv.imshow("aruco_board", im_with_aruco_board)

    # allow quitting
    if cv.waitKey(1) & 0XFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()