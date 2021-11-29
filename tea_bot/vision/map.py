import sys
import time
from threading import Thread
import cv2 as cv
from cv2 import aruco, imwrite
import numpy as np
from .calibration import CameraCalibration

CALIBRATION_FILE_DEFAULT = "share/calibration/nexigo_webcam/cfg.xml"
MARKER_LENGTH = 40 # cm
MARKER_SEPARATION = 8 # cm 

class WorldMap:
    # Aruco IDs
    TOP_LEFT_ARUCO_ID = 0
    TOP_RIGHT_ARUCO_ID = 4
    BOTTOM_LEFT_ARUCO_ID = 30
    BOTTOM_RIGHT_ARUCO_ID = 34
    
    ARM_BASE_ARUCO_ID = 1
    ARM_JOINT_ARUCO_ID = 2
    ARM_END_EFFECTOR_ARUCO_ID = 3

    # Aruco board
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    board = aruco.GridBoard_create(5, 7, MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)
    aruco_params = aruco.DetectorParameters_create()

    def __init__(self, calibration_file, video_index=0, show_feed=False) -> None:
        # Load calibration
        self.calibration = CameraCalibration(calibration_file)
        self.calibration.load_calibration()
        # Intialize camera feed
        self.vid = cv.VideoCapture(video_index)
        self.vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # Intialize state variables
        self.running = False
        self.show_feed = show_feed
        self.sensing_thread = Thread(target=self.process_image)
    
    def start(self):
        self.running = True
        self.sensing_thread.start()
    
    def stop(self):
        self.running = False

    def get_arm_coords(self):
        pass

    def get_obstacle_list(self):
        pass

    def process_image(self):
        while (self.running):
            # Capture video and undistort it
            ret,frame = self.vid.read()
            frame_undistorted = self.calibration.undistort(frame)

            # Switch to grayscale for aruco detection
            frame_undistorted_gray = cv.cvtColor(frame_undistorted, cv.COLOR_BGR2GRAY)

            # Detect aruco markers
            corners,ids,rejected = aruco.detectMarkers(frame_undistorted_gray, self.aruco_dict, parameters=self.aruco_params)
            aruco.refineDetectedMarkers(frame_undistorted_gray, self.board, corners, ids, rejected)

            if (self.show_feed):
                im_with_aruco_board = frame_undistorted
                # if we found markers draw them and see whats up with the pose board
                if ids is not None:
                    im_with_aruco_board = aruco.drawDetectedMarkers(frame_undistorted, corners, ids, (0,255,0))

                cv.imshow("world_mapping", im_with_aruco_board)
                # allow quitting
                if cv.waitKey(1) & 0XFF == ord('q'):
                    self.running = False
            
            time.sleep(0.001) # Update every 1 mSec

        # On completion destroy extra stuff
        self.vid.release()
        cv.destroyAllWindows()

def main(args):
    map = WorldMap(CALIBRATION_FILE_DEFAULT, show_feed=True)
    map.start()
    while map.running:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            map.stop()
            break

if __name__ == '__main__':
    main(sys.argv[1:])