import numpy as np
import cv2

def gaussian_filter(image, kernel_size, sigma):
    # Create a 2D Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    
    # Convolve the image with the kernel
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

# Read the image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Apply Gaussian filter
kernel_size = 5
sigma = 1.0
filtered_image = gaussian_filter(image, kernel_size, sigma)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
