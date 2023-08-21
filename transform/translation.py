import cv2
import numpy as np

def translate_image(image, dx, dy):
    # Get image dimensions
    height, width = image.shape[:2]

    # Define translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply translation to image
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

    return translated_image

# Read the input image
image = cv2.imread('image.jpg')

# Specify the translation values
dx = 50  # shift 50 pixels to the right
dy = -30  # shift 30 pixels upward

# Translate the image
translated_image = translate_image(image, dx, dy)

# Display the original and translated images
cv2.imshow('Original Image', image)
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
