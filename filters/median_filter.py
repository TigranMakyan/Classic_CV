import numpy as np
import cv2

def median_filter(image, kernel_size):
    # Pad the image to handle border pixels
    padding = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    
    # Create an empty output image
    filtered_image = np.zeros_like(image)
    
    # Apply median filtering
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest
            roi = padded_image[i:i+kernel_size, j:j+kernel_size]
            
            # Apply median operation
            median_value = np.median(roi)
            
            # Assign the median value to the corresponding pixel
            filtered_image[i, j] = median_value
    
    return filtered_image

# Read the image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Apply Median filter
kernel_size = 3
filtered_image = median_filter(image, kernel_size)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()