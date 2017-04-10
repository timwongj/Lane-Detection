from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
from src.advancedLaneDetector import AdvancedLaneDetector
from src.multiLaneDetector import MultiLaneDetector
from src.undistorter import Undistorter
from src.postprocessor import Postprocessor
from src.thresholdtypes import ThresholdTypes

# Set camera name to 'default', 'UM', or 'blackfly' for calibration and warping
camera = 'default'
# camera = 'UM'
# camera = 'blackfly'

advancedLaneDetector = AdvancedLaneDetector()
lineDetector = MultiLaneDetector()
undistorter = Undistorter(camera)
postprocessor = Postprocessor()
thresholdTypes = list(map(int, ThresholdTypes))

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

    # Define acceptable confidence
    acceptable_conf = 0.8
    left_thresh = None
    right_thresh = None

    # Lane Detection
    results = []
    for threshold in thresholdTypes:
        # Run advanced lane detection
        results.append(advancedLaneDetector.detect_lanes(
            undistorted_img, camera, threshold))
        results.append(lineDetector.detect_lanes(
            undistorted_img, camera, threshold))

        # Check if left and right confidence are acceptable
        if results[-1].left_conf >= acceptable_conf:
            left_thresh = threshold
        if results[-1].right_conf >= acceptable_conf:
            right_thresh = threshold

        # Reorder threshold priority for more efficient run on next frame
        if left_thresh is not None and right_thresh is not None:
            thresholdTypes.remove(left_thresh)
            thresholdTypes.insert(0, left_thresh)
            thresholdTypes.remove(right_thresh)
            thresholdTypes.insert(0, right_thresh)
            break

    # Post Processing
    postprocessed_image = postprocessor.postprocess(undistorted_img, results)
    misc.imsave('output_images/final.jpg', postprocessed_image)

    return postprocessed_image


if __name__ == '__main__':
    main()
