import cv2
import numpy as np

# Read the input image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Define the structuring element (kernel) for dilation
kernel = np.ones((3, 3), dtype=np.uint8)

# Apply dilation using cv2.dilate
dilated_image = cv2.dilate(image, kernel, iterations=1)
# Display the original image and dilated image
cv2.imshow('Original Image', image)
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
