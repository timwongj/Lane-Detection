import cv2
import numpy as np
from src.thresholdtypes import ThresholdTypes

SOBEL_KERNEL = 15
THRESH_DIR_MIN = 0.7
THRESH_DIR_MAX = 1.2
THRESH_MAG_MIN = 50
THRESH_MAG_MAX = 255
THRESH_YELLOW_MIN = [15, 100, 120]
THRESH_YELLOW_MIN_TOL = [15, 80, 80]
THRESH_WHITE_MIN = [0, 0, 200]
THRESH_WHITE_MIN_TOL = [0, 0, 100]


class Thresholder:
    @staticmethod
    def dir_thresh(img):
        """
        Performs direction threshold filtering
        :param img:
        :return:
        """
        sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0,
                           ksize=SOBEL_KERNEL)
        sobely = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1,
                           ksize=SOBEL_KERNEL)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= THRESH_DIR_MIN) &
                 (scaled_sobel <= THRESH_DIR_MAX)] = 255

        return sxbinary

    @staticmethod
    def mag_thresh(img):
        """
        Performs magnitude threshold filtering
        :param img:
        :return:
        """
        sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0,
                           ksize=SOBEL_KERNEL)
        sobely = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1,
                           ksize=SOBEL_KERNEL)
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= THRESH_MAG_MIN) &
                      (gradmag <= THRESH_MAG_MAX)] = 255

        return binary_output

    @staticmethod
    def color_thresh(img, y_min, w_min):
        """
        Performs white/yellow color threshold filtering
        :param img:
        :param y_min:
        :param w_min:
        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow_min = np.array(y_min, np.uint8)
        yellow_max = np.array([80, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

        white_min = np.array(w_min, np.uint8)
        white_max = np.array([255, 30, 255], np.uint8)
        white_mask = cv2.inRange(img, white_min, white_max)

        binary_output = np.zeros_like(img[:, :, 0])
        binary_output[((yellow_mask != 0) | (white_mask != 0))] = 255

        filtered = img
        filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

        return binary_output

    def combined(self, img, y_min, w_min):
        """
        Performs a combination of white/yellow color, direction, and magnitude
        threshold filtering
        :param img:
        :param y_min:
        :param w_min:
        :return:
        """
        direc = self.dir_thresh(img)
        mag = self.mag_thresh(img)
        color = self.color_thresh(img, y_min, w_min)

        combined = np.zeros_like(direc, dtype=np.uint8)
        combined[((color > 0) & ((mag > 0) | (direc > 0)))] = 255

        return combined

    @staticmethod
    def adaptive_thresh(img, type):
        """
        Performs adaptive threshold filtering
        :param img:
        :param type:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, type, cv2.THRESH_BINARY, 11, 5)
        inv_adaptive_thresh = np.zeros_like(adaptive_thresh)
        inv_adaptive_thresh[adaptive_thresh == 0] = 255
        return inv_adaptive_thresh

    @staticmethod
    def otsu_thresh(img, gaussian):
        """
        Performs otsu threshold filtering
        :param img:
        :param gaussian:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gaussian:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, otsu = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    @staticmethod
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3,
                         thresh=(0, 255)):
        """
        Performs absolute sobel threshold filtering
        :param img:
        :param orient:
        :param sobel_kernel:
        :param thresh:
        :return:
        """
        # grayscale image
        red = img[:, :, 0]

        # find abs sobel thresh
        if orient == 'x':
            sobel = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # get abs value
        abs_sobel = np.absolute(sobel)
        scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled)
        grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 255
        return grad_binary

    @staticmethod
    def hsv_thresh(img, thresh=(0, 255)):
        """
        Performs hsv threshold filtering
        :param img:
        :param thresh:
        :return:
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        v_channel = hsv[:, :, 2]

        binary_output = np.zeros_like(v_channel)
        binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 255

        return binary_output

    @staticmethod
    def hls_thresh(img, thresh=(0, 255)):
        """
        Performs hls threshold filtering
        :param img:
        :param thresh:
        :return:
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        s_channel = hls[:, :, 2]

        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255

        return binary_output

    def threshold(self, img, threshold_type):
        """
        Performs threshold filtering on img given threshold type
        :param img:
        :param threshold_type:
        :return:
        """

        if threshold_type == ThresholdTypes.COLOR:
            return self.color_thresh(img, THRESH_YELLOW_MIN, THRESH_WHITE_MIN)
        elif threshold_type == ThresholdTypes.COLOR_TOL:
            return self.color_thresh(img, THRESH_YELLOW_MIN_TOL,
                                     THRESH_WHITE_MIN_TOL)
        elif threshold_type == ThresholdTypes.ADAPT_MEAN:
            return self.adaptive_thresh(img, cv2.ADAPTIVE_THRESH_MEAN_C)
        elif threshold_type == ThresholdTypes.ADAPT_GAUSS:
            return self.adaptive_thresh(img, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        elif threshold_type == ThresholdTypes.MAG:
            return self.mag_thresh(img)
        elif threshold_type == ThresholdTypes.COMBINED:
            return self.combined(img, THRESH_YELLOW_MIN, THRESH_WHITE_MIN)
        elif threshold_type == ThresholdTypes.COMBINED_TOL:
            return self.combined(img, THRESH_YELLOW_MIN_TOL,
                                 THRESH_WHITE_MIN_TOL)
        elif threshold_type == ThresholdTypes.OTSU:
            return self.otsu_thresh(img, False)
        elif threshold_type == ThresholdTypes.OTSU_GAUSS:
            return self.otsu_thresh(img, True)
        elif threshold_type == ThresholdTypes.ABS_SOB_X:
            return self.abs_sobel_thresh(img, orient='x', sobel_kernel=3,
                                              thresh=(50, 80))
        elif threshold_type == ThresholdTypes.ABS_SOB_Y:
            return self.abs_sobel_thresh(img, orient='y', sobel_kernel=3,
                                              thresh=(50, 80))
        elif threshold_type == ThresholdTypes.ABS_SOB_X_TOL:
            return self.abs_sobel_thresh(img, orient='x', sobel_kernel=15,
                                              thresh=(30, 120))
        elif threshold_type == ThresholdTypes.ABS_SOB_Y_TOL:
            return self.abs_sobel_thresh(img, orient='x', sobel_kernel=15,
                                              thresh=(30, 120))
        elif threshold_type == ThresholdTypes.HLS:
            return self.hls_thresh(img, thresh=(100, 255))
        elif threshold_type == ThresholdTypes.HSV:
            return self.hsv_thresh(img, thresh=(50, 255))
        else:
            return self.combined(img, THRESH_YELLOW_MIN, THRESH_WHITE_MIN)
