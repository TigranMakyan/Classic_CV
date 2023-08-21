import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread('image.jpg')

# Convert the image to floating point representation
image_float = image.astype(np.float32) / 255.0

# Define the gamma value
gamma = 1.5

# Perform gamma correction
corrected_image = np.power(image_float, gamma)

# Convert the image back to 8-bit representation
corrected_image = (corrected_image * 255).astype(np.uint8)

# Display the original and corrected images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title('Gamma Corrected Image')

plt.show()
