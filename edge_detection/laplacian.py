import cv2
import numpy as np

def laplacian_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator to detect edges
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    
    # Normalize the Laplacian output to the range [0, 255]
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return laplacian

# Read the image
image = cv2.imread('image.jpg')

# Apply Laplacian edge detection
edges = laplacian_edge_detection(image)

# Display the original image and edges
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
