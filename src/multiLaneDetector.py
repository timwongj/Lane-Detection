import copy
import cv2
import numpy as np
from scipy import misc
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholder import Thresholder
from src.advancedLaneDetector import AdvancedLaneDetector
from src.warper import Warper
from src.lanechecker import Lanechecker
from src.warper import Warper

thresholder = Thresholder()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
confidence = Confidence()
laneFormatter = AdvancedLaneDetector()

class MultiLaneDetector:
    @classmethod
    def detect_lanes(self, undistorted_img, camera, threshold):
        """
        This class detects every lane and picks the innermost
        two lanes based on slope.

        :param undistorted_img: numpy matrix
        :param camera: string
        :return: AlgoResult object
        """

        # Initialize Result
        grayNorm = cv2.cvtColor(undistorted_img,cv2.COLOR_BGR2GRAY)
        misc.imsave('output_images/ld_grey.jpg', grayNorm)
        thresholdN = cv2.adaptiveThreshold(grayNorm, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        misc.imsave('output_images/ld_thN.jpg', thresholdN)
        invert = (255 - thresholdN)
        minLength = 100
        maxGap = 10
        lines = cv2.HoughLinesP(invert, 1, np.pi/180, 100,
                                lines=None,
                                minLineLength=minLength,
                                maxLineGap=maxGap)

        maxLeftSlope = 0
        maxRightSlope = 0
        self.leftLine = []
        self.rightLine = []

        # calculate slope for every line
        for line in lines:
            x1, y1, x2, y2 = line[0]
            leftSlope = self.slope(x2, x1, y1, y2)
            rightSlope = self.slope(x2, x1, y2, y1)

            # max positive slope
            if leftSlope > maxLeftSlope and undistorted_img.size / 2 > y1:
                maxLeftSlope = leftSlope
                self.leftLine = [x1, y1, x2,y2]

            # max negative slope
            if rightSlope > maxRightSlope and undistorted_img.size / 2 > y1:
                maxRightSlope = rightSlope
                self.rightLine = [x1, y1, x2, y2]

        # draw lines
        blank_image = np.zeros((invert.shape[0], invert.shape[1], 3), np.uint8)
        blank_image[:, :] = (0, 0, 0)
        cv2.line(blank_image,
                 (self.leftLine[0], self.leftLine[1]),
                 (self.leftLine[2], self.leftLine[3]),
                 (255, 255, 255), 5)
        cv2.line(blank_image,
                 (self.rightLine[0], self.rightLine[1]),
                 (self.rightLine[2], self.rightLine[3]),
                 (255, 255, 255), 5)
        misc.imsave('output_images/ld_out.jpg', blank_image)

        res = laneFormatter.detect_lanes(blank_image, camera, threshold)
        return res

    @staticmethod
    def slope(x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1)