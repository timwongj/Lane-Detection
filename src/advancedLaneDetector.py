import copy
import cv2
import numpy as np
from scipy import misc
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholder import Thresholder
from src.warper import Warper
from src.imagemerger import ImageMerger

thresholder = Thresholder()
polyfitter  = Polyfitter()
polydrawer  = Polydrawer()
confidence  = Confidence()
imagemerger = ImageMerger()

class AdvancedLaneDetector:
    def __init__(self):
        # Initialize objects to hold object data from previous frames
        self.warper = Warper()
        self.res = AlgoResult()

    def detect_lanes(self, undistorted_img, camera):
        """
        Attempts to detect left and right lanes given an undistorted image
        and camera properties
        :param undistorted_img: numpy matrix
        :param camera: string
        :return: AlgoResult object
        """

        # Merge last images together
        merged_img = imagemerger.merge(undistorted_img, 4)

        # Threshold Filtering
        img = thresholder.threshold(merged_img)
        misc.imsave('output_images/thresholded.jpg', img)

        # Warping Transformation
        self.warper.plot_trapezoid_before_warp(img)
        img = self.warper.warp(img, self.res)
        warped = copy.deepcopy(img)
        misc.imsave('output_images/warped.jpg', img)
        self.warper.plot_rectangle_after_warp(img)
        self.res.left_warp_Minv = self.warper.Minv
        self.res.right_warp_Minv = self.warper.Minv

        # Polyfit with 2nd-order interpolation
        polyfitter.plot_histogram(img)
        self.res.left_fit, self.res.right_fit = polyfitter.polyfit_sliding(img)
        self.res.left_points  = np.column_stack((polyfitter.leftx, polyfitter.lefty))
        self.res.right_points = np.column_stack((polyfitter.rightx, polyfitter.righty))

        # Compute confidence
        conf_margin = warped.shape[1] / 25
        self.res.conf, self.res.left_conf, self.res.right_conf = confidence.compute_confidence(
            warped, self.res.left_fit, self.res.right_fit, conf_margin)
        polydrawer.draw_warped_confidence(warped, self.res, conf_margin)

        # Write information
        final = copy.deepcopy(undistorted_img)
        cv2.putText(final, "Confidence: {:.2f}%".format(self.res.conf * 100),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(final, "Left conf: {:.2f}%".format(self.res.left_conf * 100),
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(final, "Right conf: {:.2f}%".format(self.res.right_conf * 100),
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw overlaid lane
        final = polydrawer.draw_lane(final, self.res)
        misc.imsave('output_images/advanced_lane_detection.jpg', final)

        return self.res
