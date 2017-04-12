import cv2
import numpy as np
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.thresholdtypes import ThresholdTypes

confidence = Confidence()
polydrawer = Polydrawer()
polyfitter = Polyfitter()


class Postprocessor:
    def postprocess(self, img, results):
        # Select left and right lanes from algorithms
        final_res = self.select_result(results)

        # Draw lines onto original image
        img = polydrawer.draw_lane(img, final_res)

        # Write information on image
        self.write_information(img, final_res)

        return img

    @staticmethod
    def select_result(results):
        # Initialize final result
        final_res = AlgoResult(None)

        # Find the highest left_conf and use its left_fit and left_warp_Minv
        for res in results:
            if res.left_conf > final_res.left_conf:
                final_res.left_conf = res.left_conf
                final_res.left_fit = res.left_fit
                final_res.left_warp_Minv = res.left_warp_Minv
                final_res.left_alg = res.left_alg
                final_res.left_thresh = res.left_thresh
                final_res.warp_src = res.warp_src
            if res.right_conf > final_res.right_conf:
                final_res.right_conf = res.right_conf
                final_res.right_fit = res.right_fit
                final_res.right_warp_Minv = res.right_warp_Minv
                final_res.right_alg = res.right_alg
                final_res.right_thresh = res.right_thresh
                final_res.warp_src = res.warp_src

        # Compute combined confidence
        final_res.conf = np.sqrt(final_res.left_conf * final_res.right_conf)

        return final_res

    @staticmethod
    def write_information(img, res):
        """
        Writes car position, lane curvature, and confidences on image
        :param img: 
        :param car_pos: 
        :param lane_curve: 
        :param res: 
        :return: 
        """

        # Define text attributes
        text_color = (255, 0, 0)
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = 2

        # Write text
        cv2.putText(img, "Confidence: {:.2f}%".format(res.conf * 100), 
                    (10, 50), text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Left conf: {:.2f}%".format(res.left_conf * 100),
                    (10, 100), text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Right conf: {:.2f}%".format(res.right_conf * 100),
                    (10, 150), text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Left Thresh: {}".format(
            ThresholdTypes(res.left_thresh).name),
                    (10, 200), text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Right Thresh: {}".format(
            ThresholdTypes(res.right_thresh).name),
                    (10, 250), text_font, 1, text_color, text_thickness)
