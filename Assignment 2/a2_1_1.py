import cv2
import numpy as np
from matplotlib import pyplot as plt

#Filter the image with given n x n matrix
def kernel_filter (img, kernel):
	height, width = img.shape
	kernel_height, kernel_width = kernel.shape

	mtx_multiply = np.zeros((kernel_height, kernel_width), dtype = np.float64)
	img_convert = np.zeros((kernel_height, kernel_width), dtype = np.float64)

	img_filter = np.zeros(((height+2), (width+2)), dtype = np.float64)

	for row in range(1, height-1):
		for col in range(1, width-1):
			mtx_multiply = img[(row - 1): (row + 2), (col - 1): (col + 2)]

			img_convert = kernel*mtx_multiply
			img_filter[row, col] = img_convert.sum()
	#return the filtered image file
	return img_filter

#Call an image and rezise it into 480x640
img = cv2.imread('https-%2F%2Fpbs.twimg.com%2Fmedia%2FCx-wU9tVIAAOLdS.jpg', cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (480, 640))
height, width = img_resized.shape
#zero padded image
img_pad = np.zeros(((height + 2), (width + 2)), dtype=np.float64)

for row in range(0, height):
    for col in range(0, width):
        img_pad[row + 1, col + 1] = img_resized[row, col]

kernel_ver = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
kernel_hor = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])
kernel_gauss = np.array([(1/16, 2/16, 1/16), (2/16, 4/16, 2/16), (1/16, 2/16, 1/16)])

img_gauss = kernel_filter(img_pad, kernel_gauss)
img_x_der = kernel_filter(img_pad, kernel_hor)
img_y_der = kernel_filter(img_pad, kernel_ver)
img_x_gauss = kernel_filter(img_x_der, kernel_gauss)
img_y_gauss = kernel_filter(img_y_der, kernel_gauss)

img_x2 = img_x_der*img_x_der
img_y2 = img_y_der*img_y_der
img_xy = img_x_der*img_y_der

g_img_x2 = kernel_filter(img_x2, kernel_gauss)
g_img_y2 = kernel_filter(img_y2, kernel_gauss)
g_img_xy = kernel_filter(img_xy, kernel_gauss)

img_harris = g_img_x2*g_img_y2 - g_img_xy*g_img_xy - 0.12*(g_img_x2+g_img_y2)*(g_img_x2+g_img_y2)

plt.subplot(), plt.imshow(img_harris, cmap = 'gray'), plt.title('HARRIS CORNER DETECTION')
plt.xticks([]), plt.yticks([])
plt.show()