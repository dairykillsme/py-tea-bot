import numpy as np
import cv2 as cv
import glob
import sys
import argparse

CAMERA_MATRIX_NAME = "cameraMatrix"
DIST_COEFFS_NAME = "distCoeffs"

class CameraCalibration:
    def __init__(self, calibration_file : str):
        self.calibration_file = calibration_file
        self.camera_matrix = np.empty((1,1))
        self.dist_coeffs = np.empty((1,1))

    def load_calibration(self):
        """Loads the camera calibration file"""
        calibration_params = cv.FileStorage(self.calibration_file, cv.FILE_STORAGE_READ)
        self.camera_matrix = calibration_params.getNode(CAMERA_MATRIX_NAME).mat()
        self.dist_coeffs = calibration_params.getNode(DIST_COEFFS_NAME).mat()

    def generate_calibration(self, calibration_images_directory : str):
        """Generates a calibration file to correct for camera warping
            source: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
        """
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(calibration_images_directory + '/*.jpg')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (7,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (7,6), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        cv.destroyAllWindows()

        # Find camera calibration parameters
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Save calibration parameters
        calibration_file = cv.FileStorage(self.calibration_file, cv.FILE_STORAGE_WRITE)
        calibration_file.write(name=CAMERA_MATRIX_NAME, val=mtx)
        calibration_file.write(name=DIST_COEFFS_NAME, val=dist)
        calibration_file.release()

        # Update class parameters
        self.camera_matrix = mtx
        self.dist_coeffs = dist

    def undistort(self, img : np.matrix):
        """Undistort an image.
        Args:
            img : 2D or 3D image matrix
        Returns:
            dst : undistorted and cropped image matrix
        """
        h,w = img.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        # undistort image
        x, y, w, h = roi
        dst = cv.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx)
        return dst[y:y+h, x:x+w]


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_file', '-f', help='File to save calibration to')
    parser.add_argument('--calibration_images_dir', '-c', help='Directiory of *.jpg images of chessboard patterns for calibration')
    parsed_args = parser.parse_args(args)

    # If being executed as main function, this script is used for calibration
    calibration = CameraCalibration(parsed_args.calibration_file)
    calibration.generate_calibration(parsed_args.calibration_images_dir)

    # Lets see how well this works
    vid = cv.VideoCapture(0)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while(True):
        ret, frame = vid.read()
        undistorted = calibration.undistort(frame)
        cv.imshow('distorted', frame)
        cv.imshow('undistored', undistorted)
        
        # allow quitting
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    
    vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])