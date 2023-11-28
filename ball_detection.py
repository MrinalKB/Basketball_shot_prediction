import cv2
import numpy as np

image = cv2.imread("Files/Ball.png")
# image=image[:900,:]
image_copy = image.copy()

imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_range = (7,153,28)
upper_rnge = (162,255,255)

mask = cv2.inRange(imageHSV,lower_range, upper_rnge)
ball_detection = cv2.bitwise_and(image,image, mask=mask)

contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image_copy,contours,-1,(0,255,0),3)
image_copy = cv2.resize(image_copy, (0,0),None, 0.6,0.6)
cv2.imshow("ball detection image",image_copy)
cv2.waitKey(0)