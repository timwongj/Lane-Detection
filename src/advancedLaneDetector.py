import copy
import cv2
from scipy import misc
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
    def detect_lanes(self, undistorted_img, camera):
        # Threshold Filtering
        img = thresholder.threshold(undistorted_img)
        misc.imsave('output_images/thresholded.jpg', img)

        # Warping Transformation
        warper = Warper(camera)
        warper.plot_trapezoid_before_warp(img)
        img = warper.warp(img)
        warped = copy.deepcopy(img)
        misc.imsave('output_images/warped.jpg', img)
        warper.plot_rectangle_after_warp(img)

        # Polyfit with 2nd-order interpolation
        polyfitter.plot_histogram(img)
        left_fit, right_fit = polyfitter.polyfit(img)

        # Compute confidence
        conf, left_conf, right_conf = confidence.get_confidence(warped, left_fit, right_fit)

        # Write information
        final = copy.deepcopy(undistorted_img)
        cv2.putText(final, "Confidence: {:.2f}%".format(conf * 100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 0, 0), thickness=2)
        cv2.putText(final, "Left conf: {:.2f}%".format(left_conf * 100), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 0, 0), thickness=2)
        cv2.putText(final, "Right conf: {:.2f}%".format(right_conf * 100), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 0, 0), thickness=2)

        # Draw overlaid lane
        final = polydrawer.draw(final, left_fit, right_fit, left_conf, right_conf, conf, warper.Minv, warper.Minv)
        misc.imsave('output_images/advanced_lane_detection.jpg', final)

        return left_fit, right_fit, left_conf, right_conf, warper.Minv
