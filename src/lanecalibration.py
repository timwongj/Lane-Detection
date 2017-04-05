import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class LaneCalibration(object):
    def __init__(self, img):
        self.mclicks = 0 # Tracks number of mouse clicks
        self.click_coords = [] # Tracks mouse click coordinates
        self.img = img.copy()

    def run(self):
        # Set mouse click event
        cv2.namedWindow('Initial src Selection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Initial src Selection', self.display_points)

        # Wait for four mouse clicks 
        while(self.mclicks < 4):
            # Display image
            self.display_image()

            # Update every 20ms
            cv2.waitKey(20) 

        # Display image and close once any key is pressed
        self.display_image()
        cv2.waitKey()
        cv2.destroyAllWindows()

        # Sort selected points left to right
        self.click_coords.sort(key=lambda row: (row[0]))

        # Fix large errors that occur when warping if points do not
        # line up horizontally. Assumes user was close.
        self.click_coords = np.asarray(self.click_coords)
        self.click_coords[3,1] = self.click_coords[0,1]
        self.click_coords[2,1] = self.click_coords[1,1]

        return self.click_coords

    def display_points(self,event,x,y,flags,param):
        # Text/circle parameters
        text = "(%d,%d)" %(x,y)
        radius = 7
        fill = -1
        pos = (x-20, y-15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.6
        color = (50,50,255)
        thickness = 1

        # Called on double click event
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.mclicks += 1
            if self.mclicks <= 4:
                cv2.circle(self.img, (x,y), radius, color, fill)
                cv2.putText(self.img, text, pos, font, size, color, thickness)
                cv2.line(self.img, (0,y), (self.img.shape[1],y), color)
                self.click_coords.append([x,y])

    def display_image(self):
        # Text parameters
        text = "Select four 'src' points, then press any key to exit."
        pos = (200,200)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 1
        color = (255,255,255)
        thickness = 2

        # Add text, show image
        cv2.putText(self.img, text, pos, font, size, color, thickness)
        cv2.imshow('Initial src Selection', self.img)


if __name__ == '__main__':
    image = cv2.imread('../output_images/thresholded.jpg')
    LaneCalibration = LaneCalibration(image)
    points = LaneCalibration.run()
    print(points)



