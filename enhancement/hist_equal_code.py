import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Calculate the histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Calculate the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)

    return equalized_image

# Read the input image in grayscale
image = cv2.imread('image.jpg', 0)

# Apply histogram equalization
equalized_image = histogram_equalization(image)

# Display the original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.show()
