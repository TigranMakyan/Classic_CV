import cv2

# Load the grayscale image
image = cv2.imread('image.jpg', 0)

# Set the threshold value
threshold = 150

# Perform thresholding segmentation
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
