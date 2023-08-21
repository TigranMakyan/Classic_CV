import numpy as np
import cv2

def box_filter(image, kernel_size):
    # Create a box kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # Apply the box filter
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

# Read the image
image = cv2.imread('image.jpg')

# Apply box filter
kernel_size = 3
filtered_image = box_filter(image, kernel_size)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
