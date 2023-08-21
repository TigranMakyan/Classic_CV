import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_gabor_filter(kernel_size, sigma, theta, lambd, gamma):
    # Generate a Gabor kernel
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma)
    return kernel

# Read the image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Define Gabor filter parameters
kernel_size = 31
sigma = 5
theta = np.pi / 4
lambd = 10
gamma = 0.5

# Create Gabor filter
gabor_kernel = create_gabor_filter(kernel_size, sigma, theta, lambd, gamma)

# Apply Gabor filter to the image
filtered_image = cv2.filter2D(image, cv2.CV_64F, gabor_kernel)

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Filtered Image')
plt.axis('off')
plt.show()
