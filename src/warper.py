import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from src.lanecalibration import LaneCalibration

transformations = {
    'default': {
        'src': [[260, 680], [580, 460], [700, 460], [1040, 680]]
    },
    'UM': {
        'src': [[560, 240], [450, 374], [860, 374], [690, 240]]
    },
    'blackfly': {
        'src': [[0, 1750], [1000, 1150], [1448, 1150], [2448, 1750]]
    },
    'blackfly_reduced': {
        'src': [[0, 500], [500, 200], [724, 200], [1224, 500]]
    }
}


class Warper:
    def __init__(self, camera):
        # Tracks number of warps that are called
        self.warp_counter = 0
        self.camera = camera

    def calculate_warp_shape(self, img, res, default):     
        """
        Calculates warp src and dst shapes.

        :param img : image to calculate shape on
        :param res : algoresults object holding previous frame lane results

        """
        # Get src trapezoid shape from lane lines if not first frame
        if self.warp_counter is not 0 and default is not False:
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
            try:
                src = transformations[self.camera]['src']
            except:
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

    def warp(self, img, res, default):
        # Get self.src and self.dst points
        self.calculate_warp_shape(img, res, default)

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
            cv2.polylines(before_warp_img, [np.array(self.src, np.int32)], True,
                          127, 3)
            return before_warp_img

    def plot_rectangle_after_warp(self, img):
        # Warp must be called once to initialize variables
        if self.warp_counter == 0:
            return img
        else:
            after_warp_img = np.copy(img)
            cv2.polylines(after_warp_img, [np.array(self.dst, np.int32)], True,
                          127, 3)
            return after_warp_img
