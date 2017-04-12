import copy
import cv2
import numpy as np
from scipy import misc
from src.thresholdtypes import ThresholdTypes


class Polydrawer:
    def draw_lane(self, img, res):
        """
        Draws left and right lane lines and polyfill overlay on img
        Also writes confidence information on img
        :param img:
        :param res:
        :return:
        """

        # Initialize lane overlay images
        lane_overlay = np.zeros_like(img)

        # Draw left lane curve
        left_lane_overlay = self.draw_lane_curve(
            img, res.left_fit, res.left_conf, res.left_warp_Minv)

        # Draw right lane curve
        right_lane_overlay = self.draw_lane_curve(
            img, res.right_fit, res.right_conf, res.right_warp_Minv)

        # Compute and draw polyfill lane overlay
        if res.left_fit is not None and res.right_fit is not None:
            # Combine left and right points
            left_pts = np.fliplr(np.array(left_lane_overlay.nonzero()[0:2]))
            right_pts = np.array(right_lane_overlay.nonzero()[0:2])
            pts = np.flipud(np.hstack((left_pts, right_pts)))
            pts = [np.transpose(np.array(pts, dtype=np.int32))]

            # Determine confidence color for polyfill
            r, g, b = self.get_color_gradient(res.conf)

            # Plot polyfill
            cv2.fillPoly(lane_overlay, pts, (r, g, b))

        # Combine the result with the original image
        img = cv2.addWeighted(img, 1, lane_overlay, 0.3, 0)
        img = cv2.addWeighted(img, 1, left_lane_overlay, 1, 0)
        img = cv2.addWeighted(img, 1, right_lane_overlay, 1, 0)

        return img

    def draw_lane_curve(self, img, fit, conf, warp_Minv):
        """
        Helper function for draw_lane
        Draws unwarped lane curve on img
        :param img:
        :param fit:
        :param conf:
        :param warp_Minv:
        :return:
        """

        lane_overlay = np.zeros_like(img)
        unwarped_pts = None

        # Compute lane overlay
        if fit is not None:
            # Create points from polyline curve
            fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
            fitx = fit[0] * fity ** 2 + fit[1] * fity + fit[2]
            pts = np.array([np.transpose(np.vstack([fitx, fity]))])

            # Determine confidence color for polyline
            r, g, b = self.get_color_gradient(conf)

            # Plot polyline
            cv2.polylines(lane_overlay,
                          [np.array(pts.astype(int))], False, (r, g, b), 15)

            # Unwarp image
            lane_overlay = cv2.warpPerspective(
                lane_overlay, warp_Minv, (img.shape[1], img.shape[0]))

        return lane_overlay

    def draw_warped_confidence(self, warped, res, conf_margin):
        """
        Draws confidence boundaries on warped image for left and right lanes
        Also writes confidence information on warped image
        :param warped:
        :param res:
        :param conf_margin:
        :return:
        """

        img_confidence = copy.deepcopy(warped)

        # Draw confidence boundaries for left and right lanes
        self.draw_conf_bounds(res.left_fit, conf_margin, img_confidence)
        self.draw_conf_bounds(res.right_fit, conf_margin, img_confidence)

        # Write information on image
        cv2.putText(img_confidence, "Confidence: {:.2f}%".format(
            res.conf * 100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color=127, thickness=2)
        cv2.putText(img_confidence, "Left conf: {:.2f}%".format(
            res.left_conf * 100), (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color=127, thickness=2)
        cv2.putText(img_confidence, "Right conf: {:.2f}%".format(
            res.right_conf * 100), (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color=127, thickness=2)

        misc.imsave('output_images/confidence_{}.jpg'.format(
            ThresholdTypes(res.left_thresh).name), img_confidence)

    @staticmethod
    def draw_conf_bounds(fit, conf_margin, img):
        """
        Helper function for draw_warped_confidence
        Draws left and right confidence boundaries for a lane curve on img
        :param fit:
        :param conf_margin:
        :param img:
        :return:
        """

        if fit is not None:
            # Compute points from left/right polylines
            fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
            fitx = fit[0] * fity ** 2 + fit[1] * fity + fit[2]

            # Compute left/right boundary points
            left_boundary_pts = [np.array([np.transpose(
                np.vstack([fitx - conf_margin, fity]))], np.int32)]
            right_boundary_pts = [np.array([np.transpose(
                np.vstack([fitx + conf_margin, fity]))], np.int32)]

            # Plot polyline confidence boundaries on image
            cv2.polylines(img, left_boundary_pts, False, 127, 2)
            cv2.polylines(img, right_boundary_pts, False, 127, 2)

    @staticmethod
    def get_color_gradient(conf):
        """
        Returns a color between red and green based on confidence
        Red represents low confidence and green represents high confidence
        :param conf:
        :return:
        """

        r = 255 if conf <= 0.5 else int(((1 - conf) * 2) * 255)
        g = 255 if conf >= 0.5 else int((conf * 2) * 255)
        b = 0

        return r, g, b