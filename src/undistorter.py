import glob

import cv2
import numpy as np


class Undistorter:
    def __init__(self, camera):
        self.camera = camera
        try:
            self.objpoints = np.load('camera_calibration/calibration_data/objpoints_{}.npy'.format(camera))
            self.imgpoints = np.load('camera_calibration/calibration_data/imgpoints_{}.npy'.format(camera))
            self.shape = tuple(np.load('camera_calibration/calibration_data/shape_{}.npy'.format(camera)))
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                                                   self.shape, None, None)
            self.hasCalibrationData = True
        except:
            self.objpoints = None
            self.imgpoints = None
            self.shape = None

        if self.objpoints is None or self.imgpoints is None or self.shape is None:
            self.find_corners()

    def find_corners(self):
        images = glob.glob('camera_calibration/calibration_images/camera_cal_{}/calibration*.jpg'.format(self.camera))
        self.hasCalibrationData = len(images) > 0
        if len(images) > 0:
            base_objp = np.zeros((6 * 9, 3), np.float32)
            base_objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
            self.objpoints = []
            self.imgpoints = []
            self.shape = None

            for imname in images:
                img = cv2.imread(imname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if self.shape is None:
                    self.shape = gray.shape[::-1]

                print('Finding chessboard corners on {}'.format(imname))
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

                if ret:
                    self.objpoints.append(base_objp)
                    self.imgpoints.append(corners)

            np.save('camera_calibration/calibration_data/objpoints_{}'.format(self.camera), self.objpoints)
            np.save('camera_calibration/calibration_data/imgpoints_{}'.format(self.camera), self.imgpoints)
            np.save('camera_calibration/calibration_data/shape_{}'.format(self.camera), self.shape)

            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                                                   self.shape, None, None)

    def undistort(self, img):
        if self.hasCalibrationData:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        else:
            return img