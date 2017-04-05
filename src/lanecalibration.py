import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class LaneCalibration(object):
    def __init__(self, image):
        self.mclicks = 0 # Tracks number of mouse clicks
        self.click_coords = [] # Tracks mouse click coordinates
        self.img = cv2.imread(image)

    def run(self):
        # Set mouse click event
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.draw_circle)

        # Wait for four mouse clicks 
        while(self.mclicks < 4):
            # Display image
            self.display_image()

            # Update every 20ms
            cv2.waitKey(20) 

        # Close image
        self.display_image()
        key = cv2.waitKey()
        if key ==3:
            cv2.destroyAllWindows()

        # Sort selected points in order from top left of image to bottom right
        self.click_coords.sort(key = lambda row: (row[1],row[0]))
        return np.asarray(self.click_coords)

    def draw_circle(self,event,x,y,flags,param):
        # Text/circle parameters
        text = "(%d,%d)" %(x,y)
        radius = 7
        fill = -1
        pos = (x+6, y+6)
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
                mouseX,mouseY = x,y
                self.click_coords.append([mouseX,mouseY])

    def display_image(self):
        # Text parameters
        text = "Select four points, then press 'Enter' to exit."
        pos = (200,200)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 1
        color = (0,0,0)
        thickness = 2

        # Add text, show image
        cv2.putText(self.img, text, pos, font, size, color, thickness)
        cv2.imshow('image', self.img)


if __name__ == '__main__':
    LaneCalibration = LaneCalibration('../output_images/undistorted.jpg')
    points = LaneCalibration.run()
    print(points)



