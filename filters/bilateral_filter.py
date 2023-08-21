import cv2

def bilateral_filter(image, d, sigma_color, sigma_space):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

# Read the image
image = cv2.imread('image.jpg')

# Apply Bilateral filter
diameter = 9
sigma_color = 75
sigma_space = 75
filtered_image = bilateral_filter(image, diameter, sigma_color, sigma_space)
# print(filtered_image.shape)
# print(image.shape)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
