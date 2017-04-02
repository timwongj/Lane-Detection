from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
from src.advancedLaneDetector import AdvancedLaneDetector
from src.undistorter import Undistorter
from src.postprocessor import Postprocessor

# Set camera name to 'default', 'UM', or 'blackfly' for calibration and warping
camera = 'default'
# camera = 'UM'
# camera = 'blackfly'

advancedLaneDetector = AdvancedLaneDetector()
undistorter = Undistorter(camera)
postprocessor = Postprocessor()

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
    results = [
        advancedLaneDetector.detect_lanes(undistorted_img, camera)
        # Add other algorithms here...
    ]

    # Post Processing
    postprocessed_image = postprocessor.postprocess(undistorted_img, results)

    return postprocessed_image


if __name__ == '__main__':
    main()
