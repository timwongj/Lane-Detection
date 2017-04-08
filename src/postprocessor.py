import cv2
import numpy as np
from src.algoresults import AlgoResult
from src.confidence import Confidence
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter

confidence = Confidence()
polydrawer = Polydrawer()
polyfitter = Polyfitter()


class Postprocessor:
    def postprocess(self, img, results):
        # Select left and right lanes from algorithms
        final_res = self.select_result(results)

        # Draw lines onto original image
        img = polydrawer.draw_lane(img, final_res)

        # Measure curvature and car position
        lane_curve, car_pos = polyfitter.measure_curvature(
            img, final_res.left_fit, final_res.right_fit)

        # Write information on image
        self.write_information(img, car_pos, lane_curve, final_res)

        return img

    @staticmethod
    def select_result(results):
        # Initialize final result
        final_res = AlgoResult()

        # Find the highest left_conf and use its left_fit and left_warp_Minv
        for res in results:
            if res.left_conf > final_res.left_conf:
                final_res.left_conf = res.left_conf
                final_res.left_fit = res.left_fit
                final_res.left_warp_Minv = res.left_warp_Minv
            if res.right_conf > final_res.right_conf:
                final_res.right_conf = res.right_conf
                final_res.right_fit = res.right_fit
                final_res.right_warp_Minv = res.right_warp_Minv

        # Compute combined confidence
        final_res.conf = np.sqrt(final_res.left_conf * final_res.right_conf)

        return final_res

    @staticmethod
    def write_information(img, car_pos, lane_curve, res):
        """
        Writes car position, lane curvature, and confidences on image
        :param img: 
        :param car_pos: 
        :param lane_curve: 
        :param res: 
        :return: 
        """

        # Format text
        if car_pos is None:
            car_pos_text = 'N/A'
        elif car_pos > 0:
            car_pos_text = '{}m right of center'.format(car_pos)
        else:
            car_pos_text = '{}m left of center'.format(abs(car_pos))

        if lane_curve is None:
            lane_curve_text = 'N/A'
        else:
            lane_curve_text = '{}m'.format(lane_curve.round())

        # Define text attributes
        text_color = (255, 0, 0)
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = 2

        # Write text
        cv2.putText(img, "Lane curve: {}".format(lane_curve_text), (10, 50),
                    text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100),
                    text_font, 1,  text_color, text_thickness)
        cv2.putText(img, "Confidence: {:.2f}%".format(res.conf * 100), 
                    (10, 150), text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Left conf: {:.2f}%".format(res.left_conf * 100),
                    (10, 200), text_font, 1, text_color, text_thickness)
        cv2.putText(img, "Right conf: {:.2f}%".format(res.right_conf * 100),
                    (10, 250), text_font, 1, text_color, text_thickness)
