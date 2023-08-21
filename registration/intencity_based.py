import cv2
import numpy as np

# Read the reference image and the image to be registered
ref_image = cv2.imread('reference_image.jpg', cv2.IMREAD_GRAYSCALE)
image_to_register = cv2.imread('image_to_register.jpg', cv2.IMREAD_GRAYSCALE)

# Create a ORB (Oriented FAST and Rotated BRIEF) detector
orb = cv2.ORB_create()

# Find keypoints and compute descriptors for the images
ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_image, None)
image_keypoints, image_descriptors = orb.detectAndCompute(image_to_register, None)

# Create a Brute-Force matcher and perform matching of descriptors
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(ref_descriptors, image_descriptors)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Select top N matches for registration
num_matches = 100
selected_matches = matches[:num_matches]

# Extract matched keypoints from both images
ref_matched_keypoints = []
image_matched_keypoints = []
for match in selected_matches:
    ref_matched_keypoints.append(ref_keypoints[match.queryIdx].pt)
    image_matched_keypoints.append(image_keypoints[match.trainIdx].pt)

# Convert matched keypoints to numpy arrays
ref_matched_keypoints = np.array(ref_matched_keypoints)
image_matched_keypoints = np.array(image_matched_keypoints)

# Perform affine transformation estimation (translation and rotation)
transformation_matrix, _ = cv2.estimateAffinePartial2D(ref_matched_keypoints, image_matched_keypoints)

# Apply the estimated transformation to the image to be registered
registered_image = cv2.warpAffine(image_to_register, transformation_matrix, (ref_image.shape[1], ref_image.shape[0]))

# Display the original reference image, image to be registered, and the registered image
cv2.imshow('Reference Image', ref_image)
cv2.imshow('Image to be Registered', image_to_register)
cv2.imshow('Registered Image', registered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
