import cv2
import numpy as np

image=cv2.imread('C:/Users/SALIH/Desktop/Codes/Python/ComputerVision/Triangle.jpg',0)

sift=cv2.SIFT()

keypoints=sift.detect(image,None)

image=cv2.drawKeypoints(image,keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT',image)
cv2.waitKey(0)
cv2.destroyAllWindows()