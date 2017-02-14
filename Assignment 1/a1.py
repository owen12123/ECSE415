import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def histogram(img, height, width):
	x_axis = np.array(range(256))
	color_address = np.zeros_like(x_axis)
	for row in range(1, height):
		for col in range(1, width):
			color_address[img[row, col]] += 1
	plt.plot(color_address)
	plt.ylabel('Number fo Pixel')
	plt.xlabel('Brightness code')
	plt.show()

def temperature(img, height, width):
    blue = np.zeros_like(img)
    green = np.zeros_like(img)
    red = np.zeros_like(img)

    for row in range(1, height):
        for col in range(1, width):
            r = g = b = 0.00

            if (img[row, col] >= 0 and img[row, col] < 52):
                r = 0
                g = 0
                b = 255.0 * (img[row, col] / 52.0)
            elif (img[row, col] >= 52 and img[row, col] < 103):
                r = 0
                g = (img[row, col] - 52.0) * 4.9
                b = 255 - (img[row, col] - 52.0) * 4.9
            elif (img[row, col] >= 103 and img[row, col] < 155):
                r = (img[row, col] - 103.0) * 4.9
                g = 255 - (img[row, col] - 52.0) * 4.9
                b = 0
            elif (img[row, col] >= 155 and img[row, col] < 207):
                r = 255
                g = 0
                b = (img[row, col] - 155.0) * 4.9
            else:
                r = 255
                g = (img[row, col] - 207) * 4.9
                b = 255
            blue[row, col] = b
            green[row, col] = g
            red[row, col] = r

    return cv2.merge((red, green, blue))

def pseudocolor(img_x, img_y, height, width):
    blue = np.zeros_like(img_x)
    green = np.zeros_like(img_x)
    red = np.zeros_like(img_x)


    for row in range(1, height):
        for col in range(1, width):
            arctan = r = g = b = 0.00
            if (img_x[row, col] == 0):
                if (img_y[row, col] == 0):
                    arctan = 0
                else:
                    arctan = 1.57
            else:
                arctan = math.atan(float(img_y[row, col]) / float(img_x[row, col]))
    
            if (arctan >= 0 and arctan < 0.314):
                r = 0
                g = 0
                b = 255.0 * (arctan / 0.314)
            elif (arctan >= 0.314 and arctan < 0.628):
                r = 0
                g = (164 * arctan - 52.0) * 4.9
                b = 255 - (164 * arctan - 52.0) * 4.9
            elif (arctan >= 0.628 and arctan < 0.942):
                r = (164 * arctan - 103.0) * 4.9
                g = 255 - (164 * arctan - 52.0) * 4.9
                b = 0
            elif (arctan >= 0.942 and arctan < 1.256):
                r = 255
                g = 0
                b = (164 * arctan - 155.0) * 4.9
            else:
                r = 255
                g = (164 * arctan - 207) * 4.9
                b = 255
            blue[row, col] = b
            green[row, col] = g
            red[row, col] = r
    return cv2.merge((red, green, blue))  # import image in gray and get shape

def binary(img, height, width):
	binary = np.zeros_like(img)
	bound = 127

	for row in range(1, height):
		for col in range(1, width):
			if (img[row, col] >= bound):
				binary[row, col] = 255
			else:
				binary[row, col] = 0
	return binary

img_gray = cv2.imread('asdfasdf.png', cv2.IMREAD_GRAYSCALE)
height, width = img_gray.shape

# kernel matrix
kernel_hor = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
kernel_ver = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])

img_filt_hor = np.zeros((height, width), dtype=np.uint8)
img_filt_ver = np.zeros((height, width), dtype=np.uint8)
img_filtered = np.zeros((height, width), dtype=np.uint8)
img_binary = np.zeros((height, width), dtype=np.uint8)

kernel_height, kernel_width = kernel_hor.shape

# matrix for zero padding
img_pad = np.zeros(((height + 2), (width + 2)), dtype=np.uint8)

# variable for later work
pixel = 0.00

# make zero padded image matrix
for row in range(0, height):
    for col in range(0, width):
        img_pad[row + 1, col + 1] = img_gray[row, col]

# matrix for filtering
mtx_multiply = np.zeros((kernel_height, kernel_width), dtype=np.uint8)
img_convert_x = np.zeros((kernel_height, kernel_width), dtype=np.uint8)
img_convert_y = np.zeros((kernel_height, kernel_width), dtype=np.uint8)

# filtering process
for row in range(1, height):
    for col in range(1, width):
        mtx_multiply = img_pad[(row - 1): (row + 2), (col - 1): (col + 2)]

        # make 3X3 matrix which is multiplication of kernel and 3X3 grayscale image
        img_convert_x = kernel_hor * mtx_multiply
        if img_convert_x.sum() > 255:
            img_filt_hor[row, col] = 255
        elif img_convert_x.sum() < 0:
            img_filt_hor[row, col] = 0
        else:
            img_filt_hor[row, col] = img_convert_x.sum()

        img_convert_y = kernel_ver * mtx_multiply
        if img_convert_y.sum() > 255:
            img_filt_ver[row, col] = 255
        elif img_convert_y.sum() < 0:
            img_filt_ver[row, col] = 0
        else:
            img_filt_ver[row, col] = img_convert_y.sum()

        # add two filtered images to generate sobel filtered image
        pixel = math.sqrt((float(img_filt_hor[row, col]) * float(img_filt_hor[row, col])) + (
            float(img_filt_ver[row, col]) * float(img_filt_ver[row, col])))
        if pixel > 255:
            pixel = 255
        elif pixel < 0:
            pixel = 0
        else:
            pixel = pixel
        img_filtered[row, col] = pixel

img_temperature = temperature(img_gray, height, width)
img_pseudocolor = pseudocolor(img_filt_hor, img_filt_ver, height, width)
img_histogram = histogram(img_gray, height, width)
img_binary = binary(img_gray, height, width)

# display images
cv2.imshow('Gray Image', img_gray)
cv2.imshow('Horizontal filter', img_filt_hor)
cv2.imshow('Vertical filter', img_filt_ver)
cv2.imshow('Pseudocolor(magnitude)', img_filtered)
cv2.imshow('Pseudocolor(angle)', img_pseudocolor)
cv2.imshow('Temperature', img_temperature)
cv2.imshow('Binary', img_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
