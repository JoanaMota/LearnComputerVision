import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

img = cv2.imread("../DATA/pennies.jpg")
plot_image(img, "Pennies")

# Apply simple algorithms first to compare the results with the Watershed algorithm
# Median Blur
img_blur = cv2.medianBlur(img, 25)
# Grayscale
img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
# Binary Threshold
ret, img_thresh = cv2.threshold(img_blur_gray, 160, 255, cv2.THRESH_BINARY_INV)
# Find Contours
contours, hierarchy = cv2.findContours(
    img_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(img_contours, contours, i, (255, 0, 0), 10)
plot_image(img_contours, "Pennies Contours", cmap_option="gray")

# So right now we don't have a nice way to segment this image or find the contours of these six separate pennies.

# WATERSHED ALGORITHM
# Step 1 - Apply Blur
img_blur = cv2.medianBlur(img, 35)
# Step 2.1 - Grayscale
img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
# Step 2.2 - Binary Threshold + OTSU
ret, img_thresh = cv2.threshold(
    img_blur_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Step 3 - Noise Removal
# In this case is not necessary but for other images it may be
kernel = np.ones((3, 3,), np.uint8)
opening = cv2.morphologyEx(
    img_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
plot_image(opening, "Pennies Opening",
           cmap_option="gray")

# Step 4 - Finding sure background area from dilation
sure_background = cv2.dilate(opening, kernel, iterations=3)
plot_image(opening, "Pennies Sure Background",
           cmap_option="gray")


# Step 5 - Finding sure foreground area
# Apply distance Transform
dist_transform = cv2.distanceTransform(sure_background, cv2.DIST_L2, 5)
plot_image(dist_transform, "Pennies Distance Transform", cmap_option="gray")

ret, sure_foreground = cv2.threshold(
    dist_transform, 0.7*dist_transform.max(), 255, 0)
plot_image(sure_foreground, "Pennies Sure Foreground",
           cmap_option="gray")

# Step 6 - Finding unknown region
sure_foreground = (np.uint8(sure_foreground))
unknown = cv2.subtract(sure_background, sure_foreground)
plot_image(unknown, "Pennies Unknown region",
           cmap_option="gray")

# Step 7 - Get Markers from connectedComponents(...)
ret, markers = cv2.connectedComponents(sure_foreground)
# we want to add one to all these labels so that the sure background is not zero, but one.
markers = markers + 1
markers[unknown == 255] = 0
plot_image(markers, "Pennies markers", cmap_option="gray")

# Step 8 - Use Markers as input for the Watershed algorithm
markers = cv2.watershed(img, markers)
# Step 9 - Get Contours
contours, hierarchy = cv2.findContours(
    markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(img_contours, contours, i, (255, 0, 0), 10)
plot_image(img_contours, "Pennies Contours with Watershed", cmap_option="gray")

plt.show()
