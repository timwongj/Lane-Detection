import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from src.lanecalibration import LaneCalibration


class Warper:
    def __init__(self):
        # Tracks number of warps that are called
        self.warp_counter = 0                               

    def calculate_warp_shape(self, img, res):     
        """
        Calculates warp src and dst shapes.

        :param img : image to calculate shape on
        :param res : algoresults object holding previous frame lane results

        """

        # Get src trapezoid shape from lane lines if not first frame
        if self.warp_counter is not 0:
            # Get lane line points
            left_lane, right_lane = res.calculate_lane_pts(img)

            if left_lane is not None and right_lane is not None:
                # Create 4 src points
                p1 = left_lane[-1]
                p2 = left_lane[0]
                p3 = right_lane[0]
                p4 = right_lane[-1]

                src = [p1, p2, p3, p4]
                self.src = np.array(src)

            else:
                self.src = self.default_src

        # Create default src values
        else:
            x1 = int(0.2 * img.shape[1]) 
            x2 = int(0.35 * img.shape[1]) 
            x3 = int(0.65 * img.shape[1]) 
            x4 = int(0.9 * img.shape[1]) 
            y1 = int(0.9 * img.shape[0]) 
            y2 = int(0.7 * img.shape[0]) 

            src = [[x1,y1], [x2,y2], [x3,y2], [x4,y1]]
            self.src = np.array(src)
            self.default_src = self.src

        # Calculate dst points
        x1 = int(0.2 * img.shape[1]) # 20%
        x2 = int(0.8 * img.shape[1]) # 80%
        y1 = img.shape[0] # Full height
        y2 = 0
        dst = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]  
        self.dst = np.array(dst)

    def shift_src(self, horiz_amount, vert_amount):
        """
        Shifts trapezoid shape.

        :param horiz_amount : number of pixels to shift horizontally
        :param vert_amount  : number of pixels to shift vertically

        """
        self.src[:,0] = self.src[:,0] + horiz_amount   
        self.src[:,1] = self.src[:,1] + vert_amount

    def rotate_src(self, angle):
        return 

    def scale_src(self, horiz_amount, vert_amount):
        """
        Shifts trapezoid shape. 

        :param horiz_amount : number of pixels to scale horizontally
        :param vert_amount  : number of pixels to scale vertically

        Negative number of pixels passed will result in shrinking,
        the trapezoid, while positive will stretch.

        """
        self.src[0:2,0] = self.src[0:2,0] - horiz_amount // 2
        self.src[2:4,0] = self.src[2:4,0] + horiz_amount // 2
        self.src[1:3,1] = self.src[1:3,1] + vert_amount  // 2
        self.src[0,1]   = self.src[0,1] - vert_amount  // 2
        self.src[3,1]   = self.src[3,1] - vert_amount  // 2

    def warp(self, img, res):
        # Get self.src and self.dst points
        self.calculate_warp_shape(img, res)

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
            cv2.polylines(before_warp_img, [self.src], True, 127,
                        3)  # (img, pts, closed, color, thickness)
            cv2.putText(before_warp_img, 'Before Warp', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        127)  # (img, text, point, font, scale, color)
            cv2.putText(before_warp_img, 'Image Size: {} x {}'.format(before_warp_img.shape[1], before_warp_img.shape[0]),
                        (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 127)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[0][0], self.src[0][1]),
                        (self.src[0][0] - 150, self.src[0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[1][0], self.src[1][1]),
                        (self.src[1][0] + 15, self.src[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[2][0], self.src[2][1]),
                        (self.src[2][0] + 15, self.src[2][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            cv2.putText(before_warp_img, '({}, {})'.format(self.src[3][0], self.src[3][1]),
                        (self.src[3][0] - 150, self.src[3][1]), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            misc.imsave('output_images/before_warp.jpg', before_warp_img)
            return before_warp_img
    
    def plot_rectangle_after_warp(self, img):
       # Warp must be called once to initialize variables
        if self.warp_counter == 0:
            return img
        else:
            after_warp_img = np.copy(img)
            cv2.polylines(after_warp_img, [self.dst], True, 127, 3)  # (img, pts, closed, color, thickness)
            cv2.putText(after_warp_img, 'After Warp', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        127)  # (img, text, point, font, scale, color)
            cv2.putText(after_warp_img, 'Image Size: {} x {}'.format(after_warp_img.shape[1], after_warp_img.shape[0]),
                        (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 127)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[0][0], self.dst[0][1]),
                        (self.dst[0][0] + 50, self.dst[0][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[1][0], self.dst[1][1]),
                        (self.dst[1][0] + 15, self.dst[1][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[2][0], self.dst[2][1]),
                        (self.dst[2][0] + 15, self.dst[2][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            cv2.putText(after_warp_img, '({}, {})'.format(self.dst[3][0], self.dst[3][1]),
                        (self.dst[3][0] + 50, self.dst[3][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, 127)
            misc.imsave('output_images/after_warp.jpg', after_warp_img)
            return after_warp_img


