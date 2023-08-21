import cv2
import numpy as np

def canny_edge_detection(image, threshold1, threshold2):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    
    return edges

# Read the image
image = cv2.imread('image.jpg')

# Apply Canny edge detection
threshold1 = 100
threshold2 = 200
edges = canny_edge_detection(image, threshold1, threshold2)
print(np.unique(np.array(edges)))
# Display the original image and edges
# cv2.imshow('Original Image', image)
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
