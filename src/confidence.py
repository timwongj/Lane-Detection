import numpy as np
from src.polydrawer import Polydrawer

polydrawer = Polydrawer()


class Confidence:
    def compute_confidence(self, warped, left_fit, right_fit, conf_margin):
        """
        Computes left and right lane confidences and combined confidence
        :param warped:
        :param left_fit:
        :param right_fit:
        :param conf_margin:
        :return:
        """

        # Compute left lane confidence
        left_conf = self.compute_lane_confidence(
            warped[:, 0:int(warped.shape[1]/2)],
            left_fit, conf_margin, 0)

        # Compute right lane confidence
        right_conf = self.compute_lane_confidence(
            warped[:, int(warped.shape[1]/2):warped.shape[1]],
            right_fit, conf_margin, int(warped.shape[1]/2))

        # Combine left and right confidences
        conf = np.sqrt(left_conf * right_conf)

        return conf, left_conf, right_conf

    @staticmethod
    def compute_lane_confidence(warped, fit, conf_margin, x_offset):
        """
        Helper method for compute_confidence
        Computes individual lane confidence based on accuracy and coverage
        :param warped:
        :param fit:
        :param conf_margin:
        :param x_offset:
        :return:
        """

        if fit is None:
            return 0

        # Nonzero represents all the points in the warped image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create points from the polyline curve
        fity = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        fitx = fit[0] * fity ** 2 + fit[1] * fity + fit[2] - x_offset

        # Sum up all the points that fall within the interval
        # Sum up number of rows that contain at least one point within
        #  the boundaries for coverage
        num_pts_in_boundaries = 0
        coverage_pts = np.zeros(warped.shape[0])
        for ind, ptx in np.ndenumerate(nonzerox):
            if np.absolute(ptx - fitx[nonzeroy[ind]]) < conf_margin:
                num_pts_in_boundaries += 1
                coverage_pts[nonzeroy[ind]] = 1

        # Determine accuracy of polyline and do not divide by 0
        num_pts = np.count_nonzero(nonzerox)
        accuracy = num_pts_in_boundaries / num_pts if num_pts > 0 else 0

        # Determine coverage of points within boundaries along the y-axis
        coverage = np.sqrt(coverage_pts.sum() / warped.shape[0])

        # Combine accuracy and coverage to obtain confidence
        conf = np.sqrt(accuracy * coverage)

        return conf
