import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class LaneCalibration(object):
    def __init__(self, video_path):
        self.mclicks = 0 # Tracks number of mouse clicks
        self.click_coords = [] # Tracks mouse click coordinates

        # Capture a frame from a video and save as img
        video_capture = cv2.VideoCapture(video_path)
        video_capture.set(cv2.CAP_PROP_POS_MSEC,1000) # Choose 1 seconds into video
        self.success, self.img = video_capture.read()

    def run(self):
        # If succesful image capture
        if self.success:
            # Set mouse click event
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.draw_circle)

            # Wait for four mouse clicks selecting the trapezoid shape
            while(self.mclicks < 4):
                cv2.putText(self.img, "Select four points or press 'Enter' to exit.", 
                                (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
                cv2.imshow('image', self.img)

                # Exit if 'Enter' pressed
                key = cv2.waitKey(20) & 0xFF
                if key == 13:
                    break

            # Close image
            cv2.destroyAllWindows()
            return np.asarray(self.click_coords)

    def draw_circle(self,event,x,y,flags,param):
        # Called on double click event
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img,(x,y),10,(255,0,0),-1)
            mouseX,mouseY = x,y
            self.mclicks += 1
            self.click_coords.append([mouseX,mouseY])

if __name__ == '__main__':
    LaneCalibrator = LaneCalibration('./project_video.mp4')
    points = LaneCalibrator.run()
    print(points)