import numpy as np
import cv2

class AlgoResult:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_conf = 0
        self.right_conf = 0
        self.conf = None
        self.left_warp_Minv = None
        self.right_warp_Minv = None

    def calculate_lane_pts(self, img):
        """
        Helper function that returns the points that make up
        the left and right lanes by using their fit coefficients.

        """
        # Construct x and y points given fits
        if self.left_fit is not None and self.right_fit is not None:
            fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
            left_fitx = self.left_fit[0] * fity ** 2 + self.left_fit[1] * fity + self.left_fit[2]
            right_fitx = self.right_fit[0] * fity ** 2 + self.right_fit[1] * fity + self.right_fit[2]

            # Combine points into a np array
            left_pts = np.transpose(np.vstack([left_fitx, fity]))
            right_pts = np.transpose(np.vstack([right_fitx, fity]))

            # Create blank image to overlay pts on
            left_lane_overlay = np.zeros_like(img)
            right_lane_overlay = np.zeros_like(img)

            # Plot polylines on the overlays
            cv2.polylines(left_lane_overlay,
                        [np.array(left_pts.astype(int))], False, (255,0,0), 15)
            cv2.polylines(right_lane_overlay,
                        [np.array(right_pts.astype(int))], False, (255,0,0), 15)

            # Unwarp the overlays with polylines
            left_lane_overlay = cv2.warpPerspective(
                    left_lane_overlay, self.left_warp_Minv, (img.shape[1], img.shape[0]))
            right_lane_overlay = cv2.warpPerspective(
                    right_lane_overlay, self.right_warp_Minv, (img.shape[1], img.shape[0]))

            # Combine left and right points
            left_pts = np.fliplr(np.array(left_lane_overlay.nonzero()[0:2]).T)
            right_pts = np.fliplr(np.array(right_lane_overlay.nonzero()[0:2]).T)

            return left_pts, right_pts

        else:
            return None, None

        