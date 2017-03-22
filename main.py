import cv2
import numpy as np
from numpy.polynomial import Polynomial as P
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
import numpy as np

from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
from undistorter import Undistorter
from warper import Warper

undistorter = Undistorter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()

PREV_LEFT_X1 = None
PREV_LEFT_X2 = None
PREV_RIGHT_X1 = None
PREV_RIGHT_X2 = None

BASE_IMG = None
CANNY_IMG = None

def main():
    # video = 'harder_challenge_video'
    # video = 'challenge_video'
    video = 'project_video'
    # video = 'UM'
    white_output = '{}_done_2.mp4'.format(video)
    clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(20, 20.025)
    # clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(30, 30.025)
    # clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(0, 0.025)
    white_output = '{}_both_algos.mp4'.format(video)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

def process_image(base):
    fig = plt.figure(figsize=(10, 8))
    i = 1

    # ADVANCED LANE DETECTION ALGORITHM

    undistorted = undistorter.undistort(base)
    misc.imsave('output_images/undistorted.jpg', undistorted)
    # i = show_image(fig, i, undistorted, 'Undistorted', 'gray')

    img = thresholder.threshold(undistorted)
    misc.imsave('output_images/thresholded.jpg', img)
    # i = show_image(fig, i, img, 'Thresholded', 'gray')

    before_warp_img = warper.before_warp(img)
    misc.imsave('output_images/before_warp.jpg', before_warp_img)

    img = warper.warp(img)
    misc.imsave('output_images/warped.jpg', img)
    # i = show_image(fig, i, img, 'Warped', 'gray')

    after_warp_img = warper.after_warp(img)
    misc.imsave('output_images/after_warp.jpg', after_warp_img)

    # polyfitter.plot_histogram(img)

    left_fit, right_fit = polyfitter.polyfit(img)

    img = polydrawer.draw(undistorted, left_fit, right_fit, warper.Minv)
    misc.imsave('output_images/final.jpg', img)
    # show_image(fig, i, img, 'Final')

    # plt.show()
    # plt.get_current_fig_manager().frame.Maximize(True)

    lane_curve, car_pos = polyfitter.measure_curvature(img)

    if car_pos > 0:
        car_pos_text = '{}m right of center'.format(car_pos)
    else:
        car_pos_text = '{}m left of center'.format(abs(car_pos))

    cv2.putText(img, "Lane curve: {}m".format(lane_curve.round()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 255, 255), thickness=2)
    cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=2)

    # show_image(fig, i, img, 'Final')
    # plt.imshow(img)
    # plt.show()

    # SIMPLE LANE DETECTION

    misc.imsave('debug_images/base.jpg', base)

    # convert to hsv (image, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    misc.imsave('debug_images/hsv.jpg', hsv)

    # apply gaussian blur (image, (kernel_size, kernel_size), 0)
    kernel_size = 3
    g_blur = cv2.GaussianBlur(hsv, (kernel_size, kernel_size), 0)

    # filter yellow and white
    yellow_min = np.array([65, 80, 80], np.uint8)
    yellow_max = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(g_blur, yellow_min, yellow_max)

    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 80, 255], np.uint8)
    white_mask = cv2.inRange(g_blur, white_min, white_max)

    c_filter = cv2.bitwise_and(g_blur, g_blur, mask=cv2.bitwise_or(yellow_mask, white_mask))

    # use canny edge detector to detect edges
    low_threshold = 30
    high_threshold = 130
    canny = cv2.Canny(c_filter, low_threshold, high_threshold)
    CANNY_IMG = canny

    # calculate region of interest
    # defining a blank mask to start with
    mask = np.zeros_like(canny)
    ysize = base.shape[0]
    xsize = base.shape[1]


    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(canny.shape) > 2:
        channel_count = canny.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(
        mask, 
        np.array([[(40, ysize), (xsize / 2, ysize / 2 + 40), (xsize / 2, ysize / 2 + 40), (xsize - 40, ysize)]], dtype=np.int32), 
        ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(canny, mask)
    
    # calculate hough lines
    lines = cv2.HoughLinesP(masked_image, i, np.pi / 90, 10, np.array([]), minLineLength=15, maxLineGap=10)
    height, width = masked_image.shape
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    thickness = 7

    global PREV_LEFT_X1, PREV_LEFT_X2, PREV_RIGHT_X1, PREV_RIGHT_X2
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    color = [255, 0, 0]

    for line in lines:
        line = line[0]
        s = (float(line[3]) - line[1]) / (float(line[2]) - line[0])

        if 0.3 > s > -0.3:
            continue

        if s < 0:
            if line[0] > img.shape[1] / 2 + 40:
                continue

            left_x += [line[0], line[2]]
            left_y += [line[1], line[3]]
            # cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [0, 0, 255], thickness)
        else:
            if line[0] < line_img.shape[1] / 2 - 40:
                continue

            right_x += [line[0], line[2]]
            right_y += [line[1], line[3]]
            # cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [255, 255, 0], thickness)

    y1 = line_img.shape[0]
    y2 = line_img.shape[0] / 2 + 90

    if len(left_x) <= 1 or len(right_x) <= 1:
        if PREV_LEFT_X1 is not None:
            cv2.line(line_img, (int(PREV_LEFT_X1), int(y1)), (int(PREV_LEFT_X2), int(y2)), color, thickness)
            cv2.line(line_img, (int(PREV_LEFT_X2), int(y1)), (int(PREV_RIGHT_X2), int(y2)), color, thickness)
        return

    left_poly = P.fit(np.array(left_x), np.array(left_y), 1)
    right_poly = P.fit(np.array(right_x), np.array(right_y), 1)

    left_x1 = (left_poly - y1).roots()
    right_x1 = (right_poly - y1).roots()

    left_x2 = (left_poly - y2).roots()
    right_x2 = (right_poly - y2).roots()

    if PREV_LEFT_X1 is not None:
        left_x1 = PREV_LEFT_X1 * 0.7 + left_x1 * 0.3
        left_x2 = PREV_LEFT_X2 * 0.7 + left_x2 * 0.3
        right_x1 = PREV_RIGHT_X1 * 0.7 + right_x1 * 0.3
        right_x2 = PREV_RIGHT_X2 * 0.7 + right_x2 * 0.3

    PREV_LEFT_X1 = left_x1
    PREV_LEFT_X2 = left_x2
    PREV_RIGHT_X1 = right_x1
    PREV_RIGHT_X2 = right_x2

    cv2.line(line_img, (int(left_x1), int(y1)), (int(left_x2), int(y2)), color, thickness)
    cv2.line(line_img, (int(right_x1), int(y1)), (int(right_x2), int(y2)), color, thickness)

    show_image(fig, 1, line_img, 'line_img')
    # plt.imshow(line_img)
    # plt.show()

    # weight image
    α=1.
    β=1.
    λ=0.
    weighted_image = cv2.addWeighted(img, α, line_img, β, λ)

    return weighted_image

def show_image(fig, i, img, title, cmap=None):
    a = fig.add_subplot(2, 2, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1


if __name__ == '__main__':
    main()
