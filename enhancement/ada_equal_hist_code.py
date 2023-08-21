import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_size=(8, 8)):
    height, width = image.shape

    # Calculate the number of tiles
    num_tiles_x = width // tile_size[1]
    num_tiles_y = height // tile_size[0]

    # Calculate the clip limit per tile
    clip_limit_per_tile = int(clip_limit * (tile_size[0] * tile_size[1]) / 256)

    # Initialize the equalized image
    equalized_image = np.zeros_like(image)

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Define the tile region
            start_y = i * tile_size[0]
            end_y = start_y + tile_size[0]
            start_x = j * tile_size[1]
            end_x = start_x + tile_size[1]

            # Extract the tile
            tile = image[start_y:end_y, start_x:end_x]

            # Apply histogram equalization to the tile
            clahe = cv2.createCLAHE(clipLimit=clip_limit_per_tile)
            equalized_tile = clahe.apply(tile)

            # Assign the equalized tile to the corresponding region in the equalized image
            equalized_image[start_y:end_y, start_x:end_x] = equalized_tile

    return equalized_image

# Read the input image in grayscale
image = cv2.imread('image.jpg', 0)

# Apply adaptive histogram equalization
equalized_image = adaptive_histogram_equalization(image)

# Display the original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.show()
