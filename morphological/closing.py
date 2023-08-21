import cv2
import numpy as np

# Read the input image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Define the structuring element (kernel) for dilation and erosion
kernel = np.ones((3, 3), dtype=np.uint8)

# Apply closing using cv2.morphologyEx
closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the original image and closed image
cv2.imshow('Original Image', image)
cv2.imshow('Closed Image', closed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
