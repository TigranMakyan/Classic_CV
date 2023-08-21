import cv2
import numpy as np

# Read the input image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Define the structuring element (kernel) for erosion and dilation
kernel = np.ones((3, 3), dtype=np.uint8)

# Apply opening using cv2.morphologyEx
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Display the original image and opened image
cv2.imshow('Original Image', image)
cv2.imshow('Opened Image', opened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
