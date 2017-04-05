import copy
import cv2
import numpy as np
from scipy import misc
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholder import Thresholder
from src.lanechecker import Lanechecker
from src.warper import Warper

thresholder = Thresholder()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
confidence = Confidence()
result = AlgoResult()

class MultiLaneDetector:
    @classmethod
    def detect_lanes(self, undistorted_img, camera):
        """
        This class detects every lane and picks the innermost
        two lanes based on slope.

        :param undistorted_img: numpy matrix
        :param camera: string
        :return: AlgoResult object
        """

        # Initialize Result
        res = AlgoResult()
        copy = undistorted_img

        # Blur the image
        blur = cv2.blur(undistorted_img, (8,8))
        misc.imsave('output_images/blurred_image.jpg', blur)

        # Convert to grayscale
        grayBlur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        grayNorm = cv2.cvtColor(undistorted_img,cv2.COLOR_BGR2GRAY)
        misc.imsave('output_images/gray_blurred.jpg', grayBlur)

        # Apply adaptive thresholding
        thresholdB = cv2.adaptiveThreshold(grayBlur, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        misc.imsave('output_images/threshold_blur.jpg', thresholdB)

        # invert
        invert = (255 - thresholdB)
        misc.imsave('output_images/inverted.jpg', invert)

        thresholdN = cv2.adaptiveThreshold(grayNorm, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

        # invert
        invert2 = (255 - thresholdN)

        misc.imsave('output_images/threshold_norm.jpg', thresholdN)

        minLength = 100
        maxGap = 10

        # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,None,minLength,maxGap)
        lines = cv2.HoughLinesP(invert2,1, np.pi/180,100,lines=None,
                                minLineLength=minLength,maxLineGap=maxGap)

        # calculate left line by taking maximum positive slope
        maxLeftSlope = 0
        maxRightSlope = 0
        leftLine = []
        rightLine = []

        # calculate slope for every line
        for line in lines:
            x1, y1, x2, y2 = line[0]
            #cv2.line(copy, (x1, y1),
            #         (x2, y2), (0, 255, 0), 5)
            #misc.imsave('output_images/all_lines.jpg', copy)

            leftSlope = self.slope(x2, x1, y1, y2)
            rightSlope = self.slope(x2, x1, y2, y1)

            # max positive slope
            if leftSlope > maxLeftSlope and undistorted_img.size / 2 > y1:
                maxLeftSlope = leftSlope
                leftLine = [x1,y1,x2,y2]

            # max negative slope
            if rightSlope > maxRightSlope and undistorted_img.size / 2 > y1:
                maxRightSlope = rightSlope
                rightLine = [x1,y1,x2,y2]

        # draw lines
        cv2.line(undistorted_img,(leftLine[0], leftLine[1]),
                 (leftLine[2],leftLine[3]),(0,255,0),5)
        cv2.line(undistorted_img, (rightLine[0], rightLine[1]),
                 (rightLine[2], rightLine[3]), (0, 255, 0), 5)
        misc.imsave('output_images/lines.jpg', undistorted_img)

        return res

    @staticmethod
    def slope(x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1)
