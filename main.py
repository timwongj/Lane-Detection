import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
from src.advancedLaneDetector import AdvancedLaneDetector
from src.undistorter import Undistorter
from src.polydrawer import Polydrawer
from src.polyfitter import Polyfitter

# Set camera name to 'default', 'UM', or 'blackfly' for calibration and warping
camera = 'default'
# camera = 'UM'
# camera = 'blackfly'

advancedLaneDetector = AdvancedLaneDetector()
undistorter = Undistorter(camera)
polyfitter = Polyfitter()
polydrawer = Polydrawer()

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
    undistorted = undistorter.undistort(img)
    misc.imsave('output_images/undistorted.jpg', undistorted)

    # Preprocessing
    preprocessedImg = preprocess(undistorted)

    # Lane Detection
    advancedLaneDetector.detect_lanes(preprocessedImg, camera)
    # other algorithms go here...

    postprocessed_image = postprocess(undistorted)

    return postprocessed_image


def preprocess(undistorted):
    # Other preprocessing goes here...
    preprocessed_image = undistorted

    return preprocessed_image


def postprocess(undistorted):
    # Confidence and Validation

    # Select left and right lanes from algorithms
    # left_line = best(all algorithms)
    # right_line = best(all algorithms)

    # Draw lines onto original image
    img = advancedLaneDetector.img
    lane_curve, car_pos = polyfitter.measure_curvature(img, advancedLaneDetector.left_fit,
                                                       advancedLaneDetector.right_fit)

    # Write information on image
    if lane_curve is not None and car_pos is not None:
        if car_pos > 0:
            car_pos_text = '{}m right of center'.format(car_pos)
        else:
            car_pos_text = '{}m left of center'.format(abs(car_pos))

        cv2.putText(img, "Lane curve: {}m".format(lane_curve.round()), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
        cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)

    return img


if __name__ == '__main__':
    main()
