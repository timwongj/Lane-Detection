import copy
import cv2
from scipy import misc
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholder import Thresholder
from src.warper import Warper

thresholder = Thresholder()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
confidence = Confidence()

class AdvancedLaneDetector:
    @staticmethod
    def detect_lanes(undistorted_img, camera):
        """
        Attempts to detect left and right lanes given an undistorted image
        and camera properties
        :param undistorted_img: numpy matrix
        :param camera: string
        :return: AlgoResult object
        """

        # Initialize Result
        res = AlgoResult()

        # Threshold Filtering
        img = thresholder.threshold(undistorted_img)
        misc.imsave('output_images/thresholded.jpg', img)

        # Warping Transformation
        warper = Warper(camera)
        res.left_warp_Minv = warper.Minv
        res.right_warp_Minv = warper.Minv
        warper.plot_trapezoid_before_warp(img)
        img = warper.warp(img)
        warped = copy.deepcopy(img)
        misc.imsave('output_images/warped.jpg', img)
        warper.plot_rectangle_after_warp(img)

        # Polyfit with 2nd-order interpolation
        polyfitter.plot_histogram(img)
        res.left_fit, res.right_fit = polyfitter.polyfit(img)

        # Compute confidence
        conf_margin = warped.shape[1] / 25
        res.conf, res.left_conf, res.right_conf = confidence.compute_confidence(
            warped, res.left_fit, res.right_fit, conf_margin)
        polydrawer.draw_warped_confidence(warped, res, conf_margin)

        # Write information
        final = copy.deepcopy(undistorted_img)
        cv2.putText(final, "Confidence: {:.2f}%".format(res.conf * 100),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(final, "Left conf: {:.2f}%".format(res.left_conf * 100),
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(final, "Right conf: {:.2f}%".format(res.right_conf * 100),
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw overlaid lane
        final = polydrawer.draw_lane(final, res)
        misc.imsave('output_images/advanced_lane_detection.jpg', final)

        return res
