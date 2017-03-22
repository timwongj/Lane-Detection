import cv2
import numpy as np

# [top-left, top-right, bottom-right, bottom-left]

# src_arr = [[580, 460], [700, 460], [1040, 680], [260, 680]] # 1280 x 720 image
# dst_arr = [[260, 0], [1040, 0], [1040, 720], [260, 720]] # 1280 x 720 image

src_arr = [[450, 374], [860, 374], [690, 240], [560, 240]] # 1242 x 374 image
dst_arr = [[260, 374], [982, 374], [982, 0], [260, 0]] # 1242 x 374 image

class Warper:
    def __init__(self):
        src = np.float32(src_arr)
        dst = np.float32(dst_arr)

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

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
    
    def before_warp(self, img):
        before_warp_img = np.copy(img)
        cv2.polylines(before_warp_img, [np.array(src_arr, np.int32)], True, 1,
                      2)  # (img, pts, closed, color, thickness)
        cv2.putText(before_warp_img, 'Before Warp', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                    1)  # (img, text, point, font, scale, color)
        cv2.putText(before_warp_img, 'Image Size: {} x {}'.format(before_warp_img.shape[1], before_warp_img.shape[0]),
                    (30, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        cv2.putText(before_warp_img, '({}, {})'.format(src_arr[0][0], src_arr[0][1]),
                    (src_arr[0][0] - 150, src_arr[0][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        cv2.putText(before_warp_img, '({}, {})'.format(src_arr[1][0], src_arr[1][1]),
                    (src_arr[1][0] + 15, src_arr[1][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        cv2.putText(before_warp_img, '({}, {})'.format(src_arr[2][0], src_arr[2][1]),
                    (src_arr[2][0] + 15, src_arr[2][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        cv2.putText(before_warp_img, '({}, {})'.format(src_arr[3][0], src_arr[3][1]),
                    (src_arr[3][0] - 150, src_arr[3][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        return before_warp_img
    
    def after_warp(self, img):
        after_warp_img = np.copy(img)
        dst = np.array(dst_arr, np.int32)
        cv2.polylines(after_warp_img, [dst], True, 1, 2)  # (img, pts, closed, color, thickness)
        cv2.putText(after_warp_img, 'After Warp', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                    1)  # (img, text, point, font, scale, color)
        cv2.putText(after_warp_img, 'Image Size: {} x {}'.format(after_warp_img.shape[1], after_warp_img.shape[0]),
                    (30, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        cv2.putText(after_warp_img, '({}, {})'.format(dst_arr[0][0], dst_arr[0][1]),
                    (dst_arr[0][0] + 50, dst_arr[0][1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        cv2.putText(after_warp_img, '({}, {})'.format(dst_arr[1][0], dst_arr[1][1]),
                    (dst_arr[1][0] + 15, dst_arr[1][1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        cv2.putText(after_warp_img, '({}, {})'.format(dst_arr[2][0], dst_arr[2][1]),
                    (dst_arr[2][0] + 15, dst_arr[2][1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        cv2.putText(after_warp_img, '({}, {})'.format(dst_arr[3][0], dst_arr[3][1]),
                    (dst_arr[3][0] + 50, dst_arr[3][1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)
        return after_warp_img