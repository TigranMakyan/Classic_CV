import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
from skimage.io import imread
from skimage.color import rgb2gray

# Load the image
image = imread('image.jpg')
image_gray = rgb2gray(image)

# Define parameters for DoG blob detection
min_sigma = 1
max_sigma = 30
num_sigma = 10
threshold = 0.01
overlap = 0.5

# Perform DoG blob detection
blobs = blob_dog(image_gray, min_sigma=min_sigma, max_sigma=max_sigma,
                 threshold=threshold, overlap=overlap)

# Display the image and detected blobs
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
for blob in blobs:
    y, x, radius = blob
    circle = plt.Circle((x, y), radius, color='r', fill=False)
    ax.add_patch(circle)

plt.title('DoG Blob Detection')
plt.axis('off')
plt.show()
