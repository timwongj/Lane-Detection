import cv2
import numpy as np
from imagemerger import ImageMerger

# Read in two test images
img1 = cv2.imread('/home/chris/Lane-Detection/test_images/test1.jpg')
img2 = cv2.imread('/home/chris/Lane-Detection/test_images/test3.jpg')

# Stack inputs side to side
input_img = np.concatenate((img1, img2), axis=1)

# Merge two images together
imagemerger = ImageMerger(2)
merged_image1 = imagemerger.merge(img1)
merged_image2 = imagemerger.merge(img2)

# Stack outputs side by side
output_img = np.concatenate((merged_image1, merged_image2), axis=1)

# Stack inputs and outputs vertically
final_img = np.concatenate((input_img, output_img), axis=0)

# Text parameters
text1 = "Input1"
text2 = "Input2"
text3 = "Output1"
text4 = "Output2"
pos1 = (500, 100)
pos2 = (1800, 100)
pos3 = (500, 800)
pos4 = (1800, 800)
font = cv2.FONT_HERSHEY_SIMPLEX
size = 2
color = (100,0,255)
thickness = 3

# Add text, show image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.putText(final_img, text1, pos1, font, size, color, thickness)
cv2.putText(final_img, text2, pos2, font, size, color, thickness)
cv2.putText(final_img, text3, pos3, font, size, color, thickness)
cv2.putText(final_img, text4, pos4, font, size, color, thickness)
cv2.imshow('image', final_img)
cv2.waitKey()


