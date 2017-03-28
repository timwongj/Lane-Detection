import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
from src.advancedLaneDetector import AdvancedLaneDetector
from src.undistorter import Undistorter
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter
from src.confidence import Confidence

# Set camera name to 'default', 'UM', or 'blackfly' for calibration and warping
camera = 'default'
# camera = 'UM'
# camera = 'blackfly'

advancedLaneDetector = AdvancedLaneDetector()
undistorter = Undistorter(camera)
polyfitter = Polyfitter()
polydrawer = Polydrawer()
confidence = Confidence()

def main():
    video = 'data/project_video'
    # video = 'data/UM'
    # video = 'data/sample14'
    white_output = '{}_done.mp4'.format(video)
    clip1 = VideoFileClip('{}.mp4'.format(video))
    clip1 = clip1.subclip(0, 0.025)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


def process_image(img):
    # Image Undistortion
    undistorted_img = undistorter.undistort(img)
    misc.imsave('output_images/undistorted.jpg', undistorted_img)

    # Lane Detection
    results = np.array([
        advancedLaneDetector.detect_lanes(undistorted_img, camera)
        # Add other algorithms here...
    ])

    # Post Processing
    postprocessed_image = postprocess(undistorted_img, results)

    return postprocessed_image


def postprocess(img, results):
    # Select left and right lanes from algorithms
    left_fit, right_fit, left_conf, right_conf, conf, left_warp_Minv, right_warp_Minv = \
        confidence.select_result(results)

    # Draw lines onto original image
    img = polydrawer.draw(img, left_fit, right_fit, left_conf, right_conf, conf, left_warp_Minv, right_warp_Minv)

    # Measure curvature and car position
    lane_curve, car_pos = polyfitter.measure_curvature(img, left_fit, right_fit)

    # Write information on image
    if car_pos is None:
        car_pos_text = 'N/A'
    elif car_pos > 0:
        car_pos_text = '{}m right of center'.format(car_pos)
    else:
        car_pos_text = '{}m left of center'.format(abs(car_pos))
    lane_curve_text = '{}m'.format(lane_curve.round()) if lane_curve is not None else 'N/A'
    cv2.putText(img, "Lane curve: {}".format(lane_curve_text), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
    cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
    cv2.putText(img, "Confidence: {:.2f}%".format(conf * 100), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 0, 0), thickness=2)
    cv2.putText(img, "Left conf: {:.2f}%".format(left_conf * 100), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 0, 0), thickness=2)
    cv2.putText(img, "Right conf: {:.2f}%".format(right_conf * 100), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 0, 0), thickness=2)

    misc.imsave('output_images/final.jpg', img)

    return img


if __name__ == '__main__':
    main()
