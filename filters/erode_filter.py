import cv2
import numpy as np

def erode_filter(image, kernel_size, iterations=1):
    # Create a structuring element
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Apply the erosion filter
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    
    return eroded_image

# Read the image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Apply erosion filter
kernel_size = 3
iterations = 1
eroded_image = erode_filter(image, kernel_size, iterations)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
