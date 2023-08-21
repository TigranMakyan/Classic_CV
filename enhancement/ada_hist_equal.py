import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image in grayscale
image = cv2.imread('image.jpg', 0)

# Apply adaptive histogram equalization using cv2.createCLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_image = clahe.apply(image)

# Display the original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.show()
