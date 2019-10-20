import cv2
import numpy as np

image=cv2.imread('C:/Users/SALIH/Desktop/Codes/Python/ComputerVision/Triangle.jpg',0)

orb = cv2.ORB(1000)

keypoints=orb.detect(image,None)

keypoints,descriptors=orb.compute(image,keypoints)

imag=cv2.drawKeypoints(image,keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('ORB',imag)
cv2.waitKey(0)
cv2.destroyAllWindows()
