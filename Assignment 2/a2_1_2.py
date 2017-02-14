import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('20170203_210437.jpg', cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (480, 640))
img2 = cv2.imread('20170203_210440.jpg', cv2.IMREAD_GRAYSCALE)
img2_resized = cv2.resize(img2, (480, 640))

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2)

# Compute SIFT descriptors
keypoints,descriptors = sift.detectAndCompute(img_resized, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_resized, None)

match = bf.match(descriptors, descriptors2)

#Match the best 50
matche = sorted(match, key = lambda x:x.distance)
img3 = cv2.drawMatches(img_resized, keypoints, img2_resized, keypoints2, matche[:100], None)

#show
cv2.imshow('match', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()