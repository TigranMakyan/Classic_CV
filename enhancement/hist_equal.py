import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image in grayscale
image = cv2.imread('image.jpg', 0)

# Apply histogram equalization using cv2.equalizeHist
equalized_image = cv2.equalizeHist(image)

# Display the original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.show()
