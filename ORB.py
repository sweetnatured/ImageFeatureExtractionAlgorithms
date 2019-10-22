import cv2
import numpy as np

image=cv2.imread('C:/Users/SALIH/Desktop/Codes/Python/ComputerVision/truck.jpg',0)
resized = cv2.resize(image, (1000,500), interpolation = cv2.INTER_AREA)

orb = cv2.ORB_create(nfeatures=100)

keypoints=orb.detect(resized,None)

keypoints,descriptors=orb.compute(resized,keypoints)

imag=cv2.drawKeypoints(resized,keypoints,outImage=None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('ORB',imag)
cv2.waitKey(0)
cv2.destroyAllWindows()
