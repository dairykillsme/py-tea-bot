import sys
import time
from threading import Thread
import cv2 as cv
from cv2 import aruco, imwrite, perspectiveTransform
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
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

    def __init__(self, calibration_file, area_width=42, area_height=42, video_index=0, show_feed=False) -> None:
        # Load calibration
        self.calibration = CameraCalibration(calibration_file)
        self.calibration.load_calibration()
        # Intialize camera feed
        self.vid = cv.VideoCapture(video_index)
        self.vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # Intialize state variables
        self.running = False
        self.ready = False
        self.show_feed = show_feed
        self.sensing_thread = Thread(target=self.process_image)
        # Initialize position variables
        self.top_left = np.zeros((1,2))
        self.top_right = np.zeros((1,2))
        self.bottom_left = np.zeros((1,2))
        self.bottom_right = np.zeros((1,2))
        self.arm_base = np.zeros((1,2))
        self.arm_joint = np.zeros((1,2))
        self.arm_end_effector = np.zeros((1,2))
        # Intialize some constants
        TOP_LEFT_SRC = [-area_width/2, area_height/2]
        TOP_RIGHT_SRC = [area_width/2, area_height/2]
        BOTTOM_LEFT_SRC = [-area_width/2, -area_height/2]
        BOTTOM_RIGHT_SRC = [area_width/2, -area_height/2]
        self.SOURCE_POINTS = np.float32([TOP_LEFT_SRC,
                                       TOP_RIGHT_SRC,
                                       BOTTOM_LEFT_SRC,
                                       BOTTOM_RIGHT_SRC])
    def start(self):
        self.running = True
        self.sensing_thread.start()
    
    def stop(self):
        self.running = False

    def get_arm_coords(self):
        pass

    def get_obstacle_list(self):
        pass

    def process_markers(self, corners, ids):
        """ Assuming all corners are fixed at known locations, arm base is 0,0,
        and all markers are coplanar (or close enough for the purposes of this)
        """
        if ids is not None:
            if len(ids) == 7:
                # Translate from camera space to world space
                for id_idx in range(0, len(ids)):
                    if ids[id_idx] == WorldMap.TOP_LEFT_ARUCO_ID:
                        self.top_left = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.TOP_RIGHT_ARUCO_ID:
                        self.top_right = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.BOTTOM_LEFT_ARUCO_ID:
                        self.bottom_left = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.BOTTOM_RIGHT_ARUCO_ID:
                        self.bottom_right = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.ARM_BASE_ARUCO_ID:
                        self.arm_base = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.ARM_JOINT_ARUCO_ID:
                        self.arm_joint = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.ARM_END_EFFECTOR_ARUCO_ID:
                        self.arm_end_effector = np.mean(corners[id_idx], axis=1)
                
                # Make arm_base the origin
                self.top_left -= self.arm_base
                self.top_right -= self.arm_base
                self.bottom_left -= self.arm_base
                self.bottom_right -= self.arm_base
                self.arm_joint -= self.arm_base
                self.arm_end_effector -= self.arm_base
                self.arm_base -= self.arm_base

                # find affine transform knowing the location of the source points
                real_points = np.float32([self.top_left[0], self.top_right[0], self.bottom_left[0], self.bottom_right[0]])
                self.warp_mat = cv.getPerspectiveTransform(real_points, self.SOURCE_POINTS)

                self.top_left = self.perspective_transform_point(self.top_left)
                self.top_right = self.perspective_transform_point(self.top_right)
                self.bottom_left = self.perspective_transform_point(self.bottom_left)
                self.bottom_right = self.perspective_transform_point(self.bottom_right)
                self.arm_joint = self.perspective_transform_point(self.arm_joint)
                self.arm_base = self.perspective_transform_point(self.arm_base)
                self.arm_end_effector = self.perspective_transform_point(self.arm_end_effector)

                if not self.ready:
                    self.ready = True
        
    def perspective_transform_point(self, point):
        q = np.dot(self.warp_mat, np.concatenate((point, [[1]]), axis=1).T)
        return np.divide(q[:2], q[2]).T

    def plot_tagged_points(self):
        # Plot points
        if (self.show_feed and self.ready):
            boundary = np.array([self.top_left,
                                 self.top_right,
                                 self.bottom_right,
                                 self.bottom_left,
                                 self.top_left])
            bx,by = boundary.T

            arm = np.array([self.arm_base,
                            self.arm_joint,
                            self.arm_end_effector])
            ax,ay = arm.T

            matplotlib.use('tkagg') # need to use different backend
            plt.cla()
            plt.plot(ax[0], ay[0])
            plt.plot(bx[0], by[0])
            plt.axis('scaled')
            plt.pause(0.0001)


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
            self.process_markers(corners, ids)

            if (self.show_feed):
                im_with_aruco_board = frame_undistorted
                # if we found markers draw them and see whats up with the pose board
                if ids is not None:
                    im_with_aruco_board = aruco.drawDetectedMarkers(frame_undistorted, corners, ids, (0,255,0))

                cv.imshow("world_mapping", im_with_aruco_board)
                # allow quitting
                if cv.waitKey(1) & 0XFF == ord('q'):
                    self.running = False

        # On completion destroy extra stuff
        self.vid.release()
        cv.destroyAllWindows()

def main(args):
    map = WorldMap(CALIBRATION_FILE_DEFAULT, show_feed=True)
    map.start()
    while map.running:
        try:
            map.plot_tagged_points()
            time.sleep(0.01)
        except KeyboardInterrupt:
            map.stop()
            break

if __name__ == '__main__':
    main(sys.argv[1:])