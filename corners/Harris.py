import cv2
import numpy as np

# Load the input image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set parameters for Harris corner detection
block_size = 2
ksize = 3
k = 0.04
threshold = 0.01

# Apply Harris corner detection
corners = cv2.cornerHarris(gray, block_size, ksize, k)
# Threshold and mark the detected corners
image[corners > threshold * corners.max()] = [0, 0, 255]

# Display the result
cv2.imshow('Harris Corner Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
