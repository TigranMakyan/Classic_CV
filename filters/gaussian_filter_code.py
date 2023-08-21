import numpy as np
import cv2
import torch


def gaussian_filter(image, kernel_size, sigma):
    # Create a 1D Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Apply the kernel to the image
    filtered_image = convolve(image, kernel)
    
    return filtered_image

def create_gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    
    return kernel

def convolve(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Pad the image to handle border pixels
    padding = kernel_height // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    
    # Create an empty output image
    filtered_image = np.zeros_like(image)
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]
            
            # Apply the kernel
            filtered_image[i, j] = np.sum(roi * kernel)
    
    return filtered_image

# Read the image
image = cv2.imread('image.jpg')  # Read as grayscale

# Apply Gaussian filter
kernel_size = 5
sigma = 1.0
filtered_image = gaussian_filter(image, kernel_size, sigma)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

