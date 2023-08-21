import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def gabor_filter(image, frequency, theta, sigma_x, sigma_y):
    # Create a Gabor kernel
    kernel = create_gabor_kernel(frequency, theta, sigma_x, sigma_y)
    
    # Apply the Gabor filter
    filtered_image = signal.convolve2d(image, kernel, mode='same', boundary='symm')
    
    return filtered_image

def create_gabor_kernel(frequency, theta, sigma_x, sigma_y):
    # Generate a grid of coordinates
    x, y = np.meshgrid(np.arange(-1, 2, 1), np.arange(-1, 2, 1))
    
    # Rotate the grid by the given angle
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # Create the Gabor kernel
    kernel = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi * frequency * x_theta)
    
    # Normalize the kernel
    kernel /= np.sum(kernel)
    print(kernel)
    return kernel

# Read the image
image = plt.imread('image.jpg')

# Convert the image to grayscale
gray_image = np.mean(image, axis=2)

# Apply Gabor filter
frequency = 0.1
theta = np.pi / 4
sigma_x = 5.0
sigma_y = 10.0
filtered_image = gabor_filter(gray_image, frequency, theta, sigma_x, sigma_y)

# Display the original and filtered images
plt.subplot(121)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()
