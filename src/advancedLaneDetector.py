import copy
from scipy import misc
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholder import Thresholder
from src.warper import Warper


class AdvancedLaneDetector:
    def __init__(self):
        self.img = None
        self.left_fit = None
        self.right_fit = None
        self.left_line = None
        self.right_line = None
        self.warped = None

    def detect_lanes(self, undistorted, camera):
        thresholder = Thresholder()
        warper = Warper(camera)
        polyfitter = Polyfitter()
        polydrawer = Polydrawer()

        img = thresholder.threshold(undistorted)
        misc.imsave('output_images/thresholded.jpg', img)

        before_warp_img = warper.before_warp(img)
        misc.imsave('output_images/before_warp.jpg', before_warp_img)

        img = warper.warp(img)
        self.warped = copy.deepcopy(img)
        misc.imsave('output_images/warped.jpg', img)

        after_warp_img = warper.after_warp(img)
        misc.imsave('output_images/after_warp.jpg', after_warp_img)

        polyfitter.plot_histogram(img)

        left_fit, right_fit = polyfitter.polyfit(img)

        img = polydrawer.draw(undistorted, left_fit, right_fit, warper.Minv)
        misc.imsave('output_images/advanced_lane_detection.jpg', img)

        self.img = img
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_line = []
        self.right_line = []
