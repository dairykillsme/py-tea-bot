import sys
import time
from threading import Thread
import cv2 as cv
from cv2 import aruco, circle, imwrite, perspectiveTransform, sepFilter2D
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
import shapely
from shapely.geometry import geo
from .calibration import CameraCalibration

CALIBRATION_FILE_DEFAULT = "share/calibration/nexigo_webcam/cfg.xml"
MARKER_LENGTH = 60 # mm
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
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    board = aruco.GridBoard_create(5, 7, MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)
    aruco_params = aruco.DetectorParameters_create()

    def __init__(self, calibration_file, area_width=36, area_height=36, video_index=0, show_feed=False) -> None:
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
        self.top_left = geometry.Point([0,0])
        self.top_right = geometry.Point([0,0])
        self.bottom_left = geometry.Point([0,0])
        self.bottom_right = geometry.Point([0,0])
        self.arm_base = geometry.Point([0,0])
        self.arm_joint = geometry.Point([0,0])
        self.arm_end_effector = geometry.Point([0,0])
        self.goal = geometry.Point([0,0])
        self.goal_detected = False
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

    def get_obstacle_list(self):
        pass

    def process_markers(self, corners, ids):
        """ Assuming all corners are fixed at known locations, arm base is 0,0,
        and all markers are coplanar (or close enough for the purposes of this)
        
        Args:
            corners - an ordered list of corners for each detected aruco marker.
            ids - an ordered list of ids for each detected aruco marker
        
        Returns:
            TRUE - all 7 markers detected
            FALSE - not enough markers found in frame
        """
        if ids is not None:
            if len(ids) == 7:
                # Translate from camera space to world space
                top_left = None
                top_right = None
                bottom_left = None
                bottom_right = None
                arm_base = None
                arm_joint = None
                arm_end_effector = None

                for id_idx in range(0, len(ids)):
                    if ids[id_idx] == WorldMap.TOP_LEFT_ARUCO_ID:
                        top_left = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.TOP_RIGHT_ARUCO_ID:
                        top_right = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.BOTTOM_LEFT_ARUCO_ID:
                        bottom_left = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.BOTTOM_RIGHT_ARUCO_ID:
                        bottom_right = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.ARM_BASE_ARUCO_ID:
                        arm_base = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.ARM_JOINT_ARUCO_ID:
                        arm_joint = np.mean(corners[id_idx], axis=1)
                    elif ids[id_idx] == WorldMap.ARM_END_EFFECTOR_ARUCO_ID:
                        arm_end_effector = np.mean(corners[id_idx], axis=1)
                
                if (top_left is None or
                    top_right is None or
                    bottom_left is None or
                    bottom_right is None or
                    arm_base is None or
                    arm_joint is None or 
                    arm_end_effector is None):
                    return False

                # find affine transform knowing the location of the source points
                real_points = np.float32([top_left[0], top_right[0], bottom_left[0], bottom_right[0]])
                self.warp_mat = cv.getPerspectiveTransform(real_points, self.SOURCE_POINTS)

                self.top_left = geometry.Point(self.perspective_transform_point(top_left)[0])
                self.top_right = geometry.Point(self.perspective_transform_point(top_right)[0])
                self.bottom_left = geometry.Point(self.perspective_transform_point(bottom_left)[0])
                self.bottom_right = geometry.Point(self.perspective_transform_point(bottom_right)[0])
                self.arm_joint = geometry.Point(self.perspective_transform_point(arm_joint)[0])
                self.arm_base = geometry.Point(self.perspective_transform_point(arm_base)[0])
                self.arm_end_effector = geometry.Point(self.perspective_transform_point(arm_end_effector)[0])

                if not self.ready:
                    self.ready = True
                
                return True
        return False
    
    def process_obstacles(self, grayscale_img):
        # Converting image to a binary image 
        # (black and white only image).
        _,threshold = cv.threshold(grayscale_img, 110, 255, 
                                    cv.THRESH_BINARY)
        # Detecting shapes in image by selecting region 
        # with same colors or intensity.
        contours,_=cv.findContours(threshold, cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        img_obstacles = []
        mapped_obstacles = []
        for cnt in contours:
            area = cv.contourArea(cnt)

            # We only care about large areas
            if area > 700 and area < 10000:
                approx = cv.approxPolyDP(cnt, 0.009*cv.arcLength(cnt, True), True)
                perspective_mapped_polygon = self.persepective_transform_polygon(approx)
                img_obstacles.append(approx)

                # convert to polygon
                shapely_poly = geometry.Polygon(np.squeeze(perspective_mapped_polygon))

                # do not add polygons that contain critical points
                if not self.contains_critical_point(shapely_poly):
                    mapped_obstacles.append(shapely_poly)

        self._img_obstacle_list = img_obstacles
        self.obstacle_list = mapped_obstacles
    
    def contains_critical_point(self, polygon):
        """Check if a polygon contains any of the critical points"""
        return (self.arm_base.within(polygon) or
                self.arm_end_effector.within(polygon) or
                self.arm_joint.within(polygon) or
                self.top_left.within(polygon) or
                self.top_right.within(polygon) or
                self.bottom_left.within(polygon) or
                self.bottom_right.within(polygon) or 
                self.goal.within(polygon))
    
    def find_goal(self, grayscale_img):
        circles = cv.HoughCircles(grayscale_img, cv.HOUGH_GRADIENT, 1.5, 1, minRadius=1, maxRadius=100)
        if circles is not None:
            boundary = geometry.Polygon([self.top_left, self.top_right, self.bottom_right, self.bottom_left])
            for circle in circles[0]:
                xy = circle[0:2]
                point = geometry.Point(self.perspective_transform_point([xy])[0])
                if boundary.contains(point):
                    self.goal = point
                    return True
        else:
            return False
        
    def perspective_transform_point(self, point):
        q = np.dot(self.warp_mat, np.concatenate((point, [[1]]), axis=1).T)
        return np.divide(q[:2], q[2]).T

    def persepective_transform_polygon(self, polygon):
        new_poly = []
        for point in polygon:
            new_poly.append(self.perspective_transform_point(point))
        return new_poly

    def plot_tagged_points(self):
        # Plot points
        if (self.show_feed and self.ready):
            boundary = np.array([self.top_left.xy,
                                 self.top_right.xy,
                                 self.bottom_right.xy,
                                 self.bottom_left.xy,
                                 self.top_left.xy])
            bx,by = boundary.T[0]

            arm = np.array([self.arm_base.xy,
                            self.arm_joint.xy,
                            self.arm_end_effector.xy])
            ax,ay = arm.T[0]

            matplotlib.use('tkagg') # need to use different backend
            plt.cla()
            plt.plot(ax, ay)
            plt.plot(bx, by)

            for polygon in self.obstacle_list:
                x,y = polygon.exterior.xy
                plt.fill(x,y,alpha=.3,fc='k',ec='none') 
            
            if self.goal_detected:
                plt.plot(self.goal.x, self.goal.y, 'ro')
                plt.text(self.goal.x, self.goal.y, 'GOAL')

            plt.axis('scaled')
            plt.xlim([-20, 20])
            plt.ylim([-20, 20])
            plt.pause(0.0001)


    def process_image(self):
        while self.running:
            # Capture video and undistort it
            ret,frame = self.vid.read()
            frame_undistorted = self.calibration.undistort(frame)

            # Switch to grayscale for aruco detection
            frame_undistorted_gray = cv.cvtColor(frame_undistorted, cv.COLOR_BGR2GRAY)

            # Detect aruco markers
            corners,ids,rejected = aruco.detectMarkers(frame_undistorted_gray, self.aruco_dict, parameters=self.aruco_params)
            aruco.refineDetectedMarkers(frame_undistorted_gray, self.board, corners, ids, rejected)
            markers_found = self.process_markers(corners, ids)
            if markers_found:
                goal_found = self.find_goal(frame_undistorted_gray)
                self.process_obstacles(frame_undistorted_gray)
                self.goal_detected = goal_found

            if self.show_feed:
                im_with_aruco_board = frame_undistorted
                # if we found markers draw them and see whats up with the pose board
                if markers_found:
                    im_with_aruco_board = aruco.drawDetectedMarkers(frame_undistorted, corners, ids, (0,255,0))
                    cv.drawContours(im_with_aruco_board, self._img_obstacle_list, -1, (0, 0, 2555), 1)
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
            if map.goal_detected and map.ready:
                map.plot_tagged_points()
            time.sleep(0.01)
        except KeyboardInterrupt:
            map.stop()
            break

if __name__ == '__main__':
    main(sys.argv[1:])