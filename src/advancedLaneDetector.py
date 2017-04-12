import copy
from scipy import misc
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholder import Thresholder
from src.thresholdtypes import ThresholdTypes
from src.warper import Warper
from src.imagemerger import ImageMerger

thresholder = Thresholder()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
confidence = Confidence()
imagemerger = ImageMerger()


class AdvancedLaneDetector:
    def __init__(self, camera):
        # Initialize objects to hold object data from previous frames
        self.warper = Warper(camera)
        self.res = AlgoResult('Advanced Lane Detection')

    def detect_lanes(self, undistorted_img, threshold_type, num_merged):
        """
        Attempts to detect left and right lanes given an undistorted image
        and camera properties
        :param undistorted_img: numpy matrix
        :param threshold_type: ThresholdTypes enum
        :param num_merged: number of images merged
        :return: AlgoResult object
        """

        # Threshold Filtering
        self.res.left_thresh = threshold_type
        self.res.right_thresh = threshold_type
        img = thresholder.threshold(undistorted_img, threshold_type)

        # Merge last images together
        img = imagemerger.merge(img, num_merged)
        self.res.num_merged_left = num_merged
        self.res.num_merged_right = num_merged

        # Warping Transformation
        self.warper.plot_trapezoid_before_warp(img)
        before_warp = self.warper.plot_trapezoid_before_warp(img)
        misc.imsave('output_images/threshold_{}.jpg'.format(
            ThresholdTypes(threshold_type).name), before_warp)
        img = self.warper.warp(img, self.res)
        warped = copy.deepcopy(img)
        self.warper.plot_rectangle_after_warp(img)
        self.res.left_warp_Minv = self.warper.Minv
        self.res.right_warp_Minv = self.warper.Minv
        self.res.warp_src = self.warper.src

        # Polyfit with 2nd-order interpolation
        self.res.left_fit, self.res.right_fit = polyfitter.polyfit_sliding(img)

        # Compute confidence
        conf_margin = warped.shape[1] / 25
        self.res.conf, self.res.left_conf, self.res.right_conf = confidence.compute_confidence(
            warped, self.res.left_fit, self.res.right_fit, conf_margin)
        polydrawer.draw_warped_confidence(warped, self.res, conf_margin)

        return copy.deepcopy(self.res)
