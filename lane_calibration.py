import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

mclicks = 0 # Tracks number of mouse clicks
trap_coords = [] # Tracks mouse click coordinates

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    global mclicks
    global trap_coords
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        mouseX,mouseY = x,y
        mclicks += 1
        trap_coords.append([mouseX,mouseY])

# Capture a frame from a video and save as img
video_path = './project_video.mp4'
video_capture = cv2.VideoCapture(video_path)
video_capture.set(cv2.CAP_PROP_POS_MSEC,1000) # Choose image at 9 seconds into video
success,img = video_capture.read()

# If succesful image capture
if success:
    # Set mouse click event
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    # Wait for four mouse clicks selecting the trapezoid shape
    while(mclicks < 4):
        cv2.putText(img, "Press 'Enter' to exit.", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        cv2.imshow('image',img)

        # Exit if 'Enter' pressed
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            break

    # Close image
    cv2.destroyAllWindows()
    print(trap_coords)