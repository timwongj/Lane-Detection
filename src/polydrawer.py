import copy
import cv2
import numpy as np
from scipy import misc


class Polydrawer:
    def draw(self, img, left_fit, right_fit, left_conf, right_conf, conf, left_warp_Minv, right_warp_Minv):
        # Initialize lane overlay images
        lane_overlay = np.zeros_like(img)
        left_lane_overlay = np.zeros_like(img)
        right_lane_overlay = np.zeros_like(img)
        left_lane_overlay_line = np.zeros_like(img[:, :, 1])
        right_lane_overlay_line = np.zeros_like(img[:, :, 1])
        left_pts_unwarped = None
        right_pts_unwarped = None

        # Recast the x and y points into usable format for cv2.fillPoly()
        fity = np.linspace(0, img.shape[0] - 1, img.shape[0])

        # Compute left lane overlay
        if left_fit is not None:
            # Create points from left polyline curve
            left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
            left_pts = np.array([np.transpose(np.vstack([left_fitx, fity]))])

            # Determine confidence color for left polyline
            r, g, b = self.get_color_gradient(left_conf)

            # Overlay temporay grayscale image with left polyline
            cv2.polylines(left_lane_overlay_line, [np.array(left_pts.astype(int))], False, 1, 1)

            # Unwarp temporary grayscale image
            left_unwarped = cv2.warpPerspective(left_lane_overlay_line, left_warp_Minv, (img.shape[1], img.shape[0]))

            # Obtain unwarped points from unwarped grayscal image
            left_pts_unwarped = np.transpose(left_unwarped).nonzero()

            # Plot colored left lane overlay on an image
            cv2.polylines(left_lane_overlay, [np.transpose(np.array([left_pts_unwarped]))], False, (r, g, b), 3)

        # Compute right lane overlay
        if right_fit is not None:
            # Create points from right polyline curve
            right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]
            right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])

            # Determine confidence color for right polyline
            r, g, b = self.get_color_gradient(right_conf)

            # Overlay temporay grayscale image with right polyline
            cv2.polylines(right_lane_overlay_line, [np.array(right_pts.astype(int))], False, 1, 1)

            # Unwarp temporary grayscale image
            right_unwarped = cv2.warpPerspective(right_lane_overlay_line, right_warp_Minv, (img.shape[1], img.shape[0]))

            # Obtain unwarped points from unwarped grayscal image
            right_pts_unwarped = np.transpose(right_unwarped).nonzero()

            # Plot colored right lane overlay on an image
            cv2.polylines(right_lane_overlay, [np.transpose(np.array([right_pts_unwarped]))], False, (r, g, b), 3)

        # Compute lane overlay
        if left_fit is not None and right_fit is not None:
            # Combine left and right points
            pts = np.hstack((left_pts_unwarped, right_pts_unwarped))
            pts = [np.transpose(np.array(pts, dtype=np.int32))]

            # Determine confidence color for polyfill
            r, g, b = self.get_color_gradient(conf)

            # Plot polyfill
            cv2.fillPoly(lane_overlay, pts, (r, g, b))

        # Combine the result with the original image
        img = cv2.addWeighted(img, 1, lane_overlay, 0.3, 0)
        img = cv2.addWeighted(img, 1, left_lane_overlay, 1, 0)
        img = cv2.addWeighted(img, 1, right_lane_overlay, 1, 0)

        return img

    @staticmethod
    def draw_warped_confidence(warped, left_fit, right_fit, conf, left_conf, right_conf, margin):
        img_confidence = copy.deepcopy(warped)

        if left_fit is not None and right_fit is not None:
            # Create points from left/right confidence boundaries of left/right polylines
            fity = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
            left_fitx_left_margin = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2] - margin
            left_fitx_right_margin = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2] + margin
            right_fitx_left_margin = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2] - margin
            right_fitx_right_margin = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2] + margin

            pts_left_left_margin = np.array([np.transpose(np.vstack([left_fitx_left_margin, fity]))], np.int32)
            pts_left_right_margin = np.array([np.transpose(np.vstack([left_fitx_right_margin, fity]))], np.int32)
            pts_right_left_margin = np.array([np.flipud(np.transpose(np.vstack([right_fitx_left_margin, fity])))],
                                             np.int32)
            pts_right_right_margin = np.array([np.flipud(np.transpose(np.vstack([right_fitx_right_margin, fity])))],
                                              np.int32)

            # Plot polyline confidence boundaries on image
            cv2.polylines(img_confidence, [pts_left_left_margin], False, 1, 2)
            cv2.polylines(img_confidence, [pts_left_right_margin], False, 1, 2)
            cv2.polylines(img_confidence, [pts_right_left_margin], False, 1, 2)
            cv2.polylines(img_confidence, [pts_right_right_margin], False, 1, 2)

        # Write information on image
        cv2.putText(img_confidence, "Confidence: {:.2f}%".format(conf * 100), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=1, thickness=2)
        cv2.putText(img_confidence, "Left conf: {:.2f}%".format(left_conf * 100), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=1, thickness=2)
        cv2.putText(img_confidence, "Right conf: {:.2f}%".format(right_conf * 100), (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=1, thickness=2)
        misc.imsave('output_images/advanced_lane_detection_confidence.jpg', img_confidence)

    @staticmethod
    def get_color_gradient(conf):
        r = 255 if conf <= 0.5 else int(((1 - conf) * 2) * 255)
        g = 255 if conf >= 0.5 else int((conf * 2) * 255)
        b = 0
        return r, g, b