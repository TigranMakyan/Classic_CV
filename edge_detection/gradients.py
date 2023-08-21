import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Sobel operator to compute gradients
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Convert the gradients to absolute values
gradient_x = np.abs(gradient_x)
gradient_y = np.abs(gradient_y)

# Optional: Convert the gradients to uint8 for visualization purposes
gradient_x = cv2.convertScaleAbs(gradient_x)
gradient_y = cv2.convertScaleAbs(gradient_y)

# Display the gradients
cv2.imshow('Gradient X', gradient_x)
cv2.imshow('Gradient Y', gradient_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
