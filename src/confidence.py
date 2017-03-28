import numpy as np
from src.polydrawer import Polydrawer

polydrawer = Polydrawer()

class Confidence:
    @staticmethod
    def get_confidence(warped, left_fit, right_fit):
        # Nonzero represents all the points in the warped image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Initialize points along y-axis
        fity = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        # Margin determines the width of the interval
        margin = warped.shape[1] / 25

        # Compute left lane confidence
        if left_fit is None:
            left_conf = 0
        else:
            # Create points from left polyline curve
            left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]

            # Sum up all the points on the left
            num_pts_left = (nonzerox < warped.shape[1]/2).sum()

            # Sum up all the points on the left that fall within the interval
            # Sum up number of rows that contain at least one point in the interval for coverage
            num_pts_left_interval = 0
            pts_left_coverage = np.zeros(warped.shape[0])
            for ind, ptx in np.ndenumerate(nonzerox):
                if ptx < warped.shape[1] / 2:
                    if np.absolute(ptx - left_fitx[nonzeroy[ind]]) < margin:
                        num_pts_left_interval += 1
                        pts_left_coverage[nonzeroy[ind]] = 1

            # Determine accuracy of left polyline and do not divide by 0
            left_accuracy = num_pts_left_interval / num_pts_left if num_pts_left > 0 else 0

            # Determine coverage of points within margin for left polyline along the y-axis
            left_coverage = np.sqrt(pts_left_coverage.sum() / warped.shape[0])

            # Combine accuracy and coverage to obtain confidence
            left_conf = np.sqrt(left_accuracy * left_coverage)

        # Compute right lane confidence
        if right_fit is None:
            right_conf = 0
        else:
            # Create points from right polyline curve
            right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

            # Sum up all the points on the left
            num_pts_right = (nonzerox >= warped.shape[1] / 2).sum()

            # Sum up all the points on the right that fall within the interval
            # Sum up number of rows that contain at least one point in the interval for coverage
            num_pts_right_interval = 0
            pts_right_coverage = np.zeros(warped.shape[0])
            for ind, ptx in np.ndenumerate(nonzerox):
                if ptx >= warped.shape[1] / 2:
                    if np.absolute(ptx - right_fitx[nonzeroy[ind]]) < margin:
                        num_pts_right_interval += 1
                        pts_right_coverage[nonzeroy[ind]] = 1

            # Determine accuracy of right polyline and do not divide by 0
            right_accuracy = num_pts_right_interval / num_pts_right if num_pts_right > 0 else 0

            # Determine coverage of points within margin for right polyline along the y-axis
            right_coverage = np.sqrt(pts_right_coverage.sum() / warped.shape[0])

            # Combine accuracy and coverage to obtain confidence
            right_conf = np.sqrt(right_accuracy * right_coverage)

        # Combine left and right confidences
        conf = np.sqrt(left_conf * right_conf)

        # Draw confidence on warped image
        polydrawer = Polydrawer()
        polydrawer.draw_warped_confidence(warped, left_fit, right_fit, conf, left_conf, right_conf, margin)

        return conf, left_conf, right_conf

    @staticmethod
    def select_result(results):
        # Find the highest left_conf and its index
        left_conf = results[:, 2].max()
        left_conf_index = np.where(results[:, 2] == left_conf)[0][0]

        # Use its corresponding left_fit and save its inverse transformation for unwarping
        left_fit = results[:, 0][left_conf_index]
        left_warp_Minv = results[:, 4][left_conf_index]

        # Find the highest right_conf and its index
        right_conf = results[:, 3].max()
        right_conf_index = np.where(results[:, 3] == right_conf)[0][0]

        # Use its corresponding right_fit and save its inverse transformation for unwarping
        right_fit = results[:, 1][right_conf_index]
        right_warp_Minv = results[:, 4][right_conf_index]

        # Compute combined confidence
        conf = np.sqrt(left_conf * right_conf)

        return left_fit, right_fit, left_conf, right_conf, conf, left_warp_Minv, right_warp_Minv
