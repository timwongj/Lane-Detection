import cv2
import numpy as np
from scipy import misc
from src.lanecalibration import LaneCalibration

class Warper:
    def __init__(self):
        # Tracks number of warps that are called
        self.warp_counter = 0

        # src is the trapezoidal road shape to focus on
        self.src = 0

        # dst is the rectangular birds eye shape to transform to
        self.dst = 0

    def calculate_warp_shape(self, img, warp_counter):
        # Calculate src points, user selects if first time
        if warp_counter == 0:
            lanecalibrator = LaneCalibration(img)
            self.src = lanecalibrator.run()

        else:
            # Adjust for curving and widening lane lines
            pass

        # Calculate dst points
        x1 = int(0.2 * img.shape[1]) # 20%
        x2 = int(0.8 * img.shape[1]) # 80%
        y1 = 0
        y2 = img.shape[0] # Full height
        dst = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]

        # Sort points by x axis
        dst.sort(key=lambda axis: (axis[0]))
        self.dst = np.array(dst)

        # Swap first two rows for proper plotting
        dst_copy = self.dst.copy()
        self.dst[0] = dst_copy[1]
        self.dst[1] = dst_copy[0]

    def shift_src(self, direction):
        return 1

    def rotate_src(self, angle):
        return 1

    def scale_src(self, horizonal, vertical):
        return 1

    def warp(self, img):
        # Get self.src and self.dst points
        self.calculate_warp_shape(img, self.warp_counter)

        # Get transform matrices
        self.M = cv2.getPerspectiveTransform(np.float32(self.src), np.float32(self.dst))
        self.Minv = cv2.getPerspectiveTransform(np.float32(self.dst), np.float32(self.src))

        # Return warped image
        self.warp_counter += 1
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        # Only unwarp an already warped image
        if self.warp_counter == 0:
            return img       
        else:
            return cv2.warpPersective(
                img,
                self.Minv,
                (img.shape[1], img.shape[0]),
                flags=cv2.INTER_LINEAR
            )
    
    def plot_trapezoid_before_warp(self, img):
        # Warp must be called once to initialize variables
        if self.warp_counter == 0:
            return img
        else:
            before_warp_img = np.copy(img)
            cv2.polylines(before_warp_img, [self.src], True, 1,
                        2)  # (img, pts, closed, color, thickness)
            cv2.putText(before_warp_img, 'Before Warp', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        1)  # (img, text, point, font, scale, color)
            cv2.putText(before_warp_img, 'Image Size: {} x {}'.format(before_warp_img.shape[1], before_warp_img.shape[0]),
                        (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[0][0], self.src[0][1]),
                        (self.src[0][0] - 150, self.src[0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[1][0], self.src[1][1]),
                        (self.src[1][0] + 15, self.src[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[2][0], self.src[2][1]),
                        (self.src[2][0] + 15, self.src[2][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[3][0], self.src[3][1]),
                        (self.src[3][0] - 150, self.src[3][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            misc.imsave('output_images/before_warp.jpg', before_warp_img)
            return before_warp_img
    
    def plot_rectangle_after_warp(self, img):
       # Warp must be called once to initialize variables
        if self.warp_counter == 0:
            return img
        else:
            after_warp_img = np.copy(img)
            cv2.polylines(after_warp_img, [self.dst], True, 1, 2)  # (img, pts, closed, color, thickness)
            cv2.putText(after_warp_img, 'After Warp', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        1)  # (img, text, point, font, scale, color)
            cv2.putText(after_warp_img, 'Image Size: {} x {}'.format(after_warp_img.shape[1], after_warp_img.shape[0]),
                        (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[0][0], self.dst[0][1]),
                        (self.dst[0][0] + 50, self.dst[0][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[1][0], self.dst[1][1]),
                        (self.dst[1][0] + 15, self.dst[1][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[2][0], self.dst[2][1]),
                        (self.dst[2][0] + 15, self.dst[2][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[3][0], self.dst[3][1]),
                        (self.dst[3][0] + 50, self.dst[3][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
            misc.imsave('output_images/after_warp.jpg', after_warp_img)
            return after_warp_img


