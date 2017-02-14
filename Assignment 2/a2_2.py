import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Q2 Celebrity facematching
# Compute LBP Histogram of img
def histAndlbp(img):
    height, width = img.shape
    img_lbp = np.zeros(img.shape)
    array_hist = np.zeros(256)
    for row in range(0, height):
        for col in range(0, width):
            img_test = np.zeros((3, 3))
            for i in range(0, 3):
                for j in range(0, 3):
                    # Binary value is always zero when outside of image range
                    if (row + (i - 1) >= height) or (row + (i - 1) < 0) or (col + (j - 1) >= width) or (
                            col + (j - 1) < 0):
                        img_test[i, j] = 0
                    # Fills lbp_ker with binary values for pixel (row,col)
                    elif (img[row + (i - 1), col + (j - 1)] >= img[row, col]):
                        img_test[i, j] = 1
                    else:
                        img_test[i, j] = 0
            # Set value for each pixel to decimal value extracted from lbp_ker
            img_lbp[row, col] = np.int(
                img_test[1, 0] * 128 + img_test[2, 0] * 64 + img_test[2, 1] * 32 + img_test[2, 2] * 16 + img_test[
                    1, 2] * 8 + img_test[0, 2] * 4 + img_test[0, 1] * 2 + img_test[0, 0])
            array_hist[np.int(img_lbp[row, col])] = array_hist[np.int(img_lbp[row, col])] + 1
    # Normalize histogram data
    array_hist = array_hist / (np.linalg.norm(array_hist))
    return array_hist


# Computes LBP Descriptor of img by splitting image in 7x7 cell array
def lbp_descriptor(img):
    #print(type(img))
    #cv2.imshow('asdfasdf', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows

    height, width = img.shape
    img_des = np.zeros((7, 7, 256))
    # Calculate historgram for all but last 18 x 21 box
    for cell_row in range(0, 6):
        for cell_col in range(0, 5):
            part_img = np.zeros((18, 21))
            part_img = img[cell_row * 18:(cell_row + 1) * 18, cell_col * 21:(cell_col + 1) * 21]
            img_des[cell_row, cell_col] = histAndlbp(part_img)
    # Calculate histogram for last box
    for sliver_row in range(0, 6):
        sliver_temp = np.zeros((18, 2))
        sliver_temp = img[sliver_row * 18:(sliver_row + 1) * 18, 127:128]
        img_des[sliver_row, 6] = histAndlbp(sliver_temp)

    return img_des


# Computes descriptor of all imgs and finds best match
def face_match(img):
    # Compute selfie descriptor
    myface_des = lbp_descriptor(img)
    best_match = '0'
    i = 0
    small_sqrtdif = 1000000000
    # Compute and compare celebrity descriptor
    for dirname, dirnames, filenames in os.walk('/Users/lordent/Desktop/School/Python/Practice/OpenCV/Ass2/img_align_celeba/img_sample'):
        # print path to all filenames.
        for filename in filenames:
            #print('file name is '+filename)
            temp_comp = cv2.imread('/Users/lordent/Desktop/School/Python/Practice/OpenCV/Ass2/img_align_celeba/img_sample/' + filename, cv2.IMREAD_GRAYSCALE)
            temp_des = lbp_descriptor(temp_comp)
            sqrtdif = 0

            # Compute sum of squared difference for every image
            for cell_row in range(0, 6):
                for cell_col in range(0, 6):
                    for hist_pos in range(0, 255):
                        sqrtdif = sqrtdif + (
                        (myface_des[cell_row, cell_col, hist_pos] - temp_des[cell_row, cell_col, hist_pos]) ** 2)
            if (sqrtdif < small_sqrtdif):
                small_sqrtdif = sqrtdif
                best_match = filename

    display_image = cv2.imread('/Users/lordent/Desktop/School/Python/Practice/OpenCV/Ass2/img_align_celeba/img_sample/' + best_match)
    cv2.imshow("Imput image", img)
    cv2.imshow("Matching with", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('202594.jpg', cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (128, 128))
face_match(img_resized)

