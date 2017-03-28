import cv2
import numpy as np
from scipy import misc

transformations = {
    'default': {
        'src': [[580, 460], [700, 460], [1040, 680], [260, 680]],
        'dst': [[260, 0], [1040, 0], [1040, 720], [260, 720]],
        'shape': [1280, 720]
    },
    'UM': {
        'src': [[450, 374], [860, 374], [690, 240], [560, 240]],
        'dst': [[260, 374], [982, 374], [982, 0], [260, 0]],
        'shape': [1242, 374]
    },
    'blackfly': {
        'src': [[1149, 1024], [1299, 1024], [2448, 1798], [0, 1798]],
        'dst': [[500, 0], [1948, 0], [1948, 2048], [500, 2048]],
        'shape': [2448, 2048]
    }
}

class Warper:
    def __init__(self, camera):
        self.src = transformations[camera]['src']
        self.dst = transformations[camera]['dst']

        self.M = cv2.getPerspectiveTransform(np.float32(self.src), np.float32(self.dst))
        self.Minv = cv2.getPerspectiveTransform(np.float32(self.dst), np.float32(self.src))

    def warp(self, img):
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        return cv2.warpPersective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
    
    def plot_trapezoid_before_warp(self, img):
        before_warp_img = np.copy(img)
        cv2.polylines(before_warp_img, [np.array(self.src, np.int32)], True, 1,
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
        after_warp_img = np.copy(img)
        dst = np.array(self.dst, np.int32)
        cv2.polylines(after_warp_img, [dst], True, 1, 2)  # (img, pts, closed, color, thickness)
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
