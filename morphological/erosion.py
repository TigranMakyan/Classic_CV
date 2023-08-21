import numpy as np

def erosion(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    
    # Create an empty output image
    eroded_image = np.zeros_like(image, dtype=np.uint8)
    
    # Apply erosion operation
    for i in range(padding_height, height - padding_height):
        for j in range(padding_width, width - padding_width):
            if np.all(image[i-padding_height:i+padding_height+1, j-padding_width:j+padding_width+1] == kernel):
                eroded_image[i, j] = 255
    
    return eroded_image

# Define the input binary image and structuring element (kernel)
image = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.uint8)

# Apply erosion operation
eroded_image = erosion(image, kernel)

# Display the original image and eroded image
print("Original Image:\n", image)
print("\nEroded Image:\n", eroded_image)
