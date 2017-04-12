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
from src.thresholdtypes import ThresholdTypes
from src.lanechecker import Lanechecker

thresholder = Thresholder()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
confidence = Confidence()
laneFormatter = AdvancedLaneDetector()

class MultiLaneDetector:
    def __init__(self):
        self.left_fit = []
        self.right_fit = []
        self.leftx = []
        self.lefty = []
        self.rightx = []
        self.righty = []

    @classmethod
    def detect_lanes(self, undistorted_img, camera, threshold):
        """
        This class detects every lane and picks the innermost
        two lanes based on slope.

        :param undistorted_img: numpy matrix
        :param camera: string
        :return: AlgoResult object
        """
        res = AlgoResult('Simple')

        grayNorm = cv2.cvtColor(undistorted_img,cv2.COLOR_BGR2GRAY)
        misc.imsave('output_images/ld_grey.jpg', grayNorm)
        thresholdN = cv2.adaptiveThreshold(grayNorm, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        misc.imsave('output_images/ld_thN.jpg', thresholdN)
        invert = (255 - thresholdN)
        misc.imsave('output_images/ld_inverted.jpg', invert)

        combined = thresholder.threshold(undistorted_img,
                                         ThresholdTypes.COMBINED)
        misc.imsave('output_images/ld_combined.jpg', combined)

        blur = cv2.blur(invert,(10,10))
        misc.imsave('output_images/ld_inverted_blurred.jpg', blur)

        minLength = 100
        maxGap = 10
        lines = cv2.HoughLinesP(invert, 1, np.pi/180, 100,
                                lines=None,
                                minLineLength=minLength,
                                maxLineGap=maxGap)

        maxLeftSlope = 0
        maxRightSlope = 0
        res.left_line = []
        res.right_line = []

        # calculate slope for every line
        for line in lines:
            x1, y1, x2, y2 = line[0]
            leftSlope = self.slope(x2, x1, y1, y2)
            rightSlope = self.slope(x2, x1, y2, y1)

            # max positive slope
            if leftSlope > maxLeftSlope:  #and undistorted_img.shape[0] / 2 > y1:
                maxLeftSlope = leftSlope
                res.left_line = [x1, y1, x2,y2]

            # max negative slope
            if rightSlope > maxRightSlope:  #and undistorted_img.shape[0] / 2 > y1:
                maxRightSlope = rightSlope
                res.right_line = [x1, y1, x2, y2]

        # draw lines
        blank_image = np.zeros((invert.shape[0], invert.shape[1], 3), np.uint8)
        blank_image[:, :] = (0, 0, 0)
        cv2.line(blank_image,
                 (res.left_line[0], res.left_line[1]),
                 (res.left_line[2], res.left_line[3]),
                 (255, 255, 255), 5)
        cv2.line(blank_image,
                 (res.right_line[0], res.right_line[1]),
                 (res.right_line[2], res.right_line[3]),
                 (255, 255, 255), 5)
        cv2.line(undistorted_img,
                 (res.left_line[0], res.left_line[1]),
                 (res.left_line[2], res.left_line[3]),
                 (0, 255, 0), 5)
        cv2.line(undistorted_img,
                 (res.right_line[0], res.right_line[1]),
                 (res.right_line[2], res.right_line[3]),
                 (0, 255, 0), 5)
        misc.imsave('output_images/ld_out.jpg', blank_image)
        misc.imsave('output_images/ld_line_ud.jpg', undistorted_img)


        # warp image
        warper = Warper(camera)
        warped_pic = warper.warp(combined)
        misc.imsave('output_images/ld_warped.jpg', warped_pic)

        # warp lines
        warped_lines = warper.warp(blank_image)
        misc.imsave('output_images/ld_warped_lines.jpg', warped_lines)

        # calculate left fit and right fit
        nonzero = np.nonzero(warped_lines)
        nonzerox, nonzeroy = nonzero[1], nonzero[0]
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for index, item in enumerate(nonzerox):
            if item < warped_lines.shape[1] / 2:
                left_x.append(item)
                left_y.append(nonzeroy[index])
            else:
                right_x.append(item)
                right_y.append(nonzeroy[index])

        # left_x, left_y = nonzero[1] < warped_lines.shape[1], nonzero[0]
        # right_x, right_y = nonzero[1] >= warped_lines.shape[1], nonzero[0]

        res.left_warp_Minv = warper.Minv
        res.right_warp_Minv = warper.Minv
        res.left_fit = np.polyfit(left_y, left_x, 2) if len(
            left_y) > 0 else None
        res.right_fit = np.polyfit(right_y, right_x, 2) if len(
            right_y) > 0 else None
        polyfitter.plot_histogram(warped_lines)


        # compute confidence

        conf_margin = warped_lines.shape[1] / 25
        res.conf, res.left_conf, res.right_conf = confidence.compute_confidence(
            warped_lines, res.left_fit, res.right_fit, conf_margin
        )

        polydrawer.draw_warped_confidence(warped_lines, res, conf_margin)

        return res

    @staticmethod
    def slope(x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1)