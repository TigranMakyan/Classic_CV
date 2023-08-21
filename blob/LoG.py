import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.io import imread
from skimage.color import rgb2gray

# Load the image
image = imread('image.jpg')
image_gray = rgb2gray(image)

# Define parameters for LoG blob detection
min_sigma = 1
max_sigma = 30
num_sigma = 10
threshold = 0.02
overlap = 0.5

# Perform LoG blob detection
blobs = blob_log(image_gray, min_sigma=min_sigma, max_sigma=max_sigma,
                 num_sigma=num_sigma, threshold=threshold, overlap=overlap)

# Display the image and detected blobs
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
for blob in blobs:
    y, x, radius = blob
    circle = plt.Circle((x, y), radius, color='r', fill=False)
    ax.add_patch(circle)

plt.title('LoG Blob Detection')
plt.axis('off')
plt.show()
