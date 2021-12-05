import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

img = cv2.imread("../DATA/rainbow.jpg", 0)  # read in grayscale

plot_image(img, "Rainbow Image", cmap_option="gray")
print(img.min())
print(img.max())

# All values bellow return 0 and above return 1 in case of binary
ret, img_threshold = cv2.threshold(
    img, thresh=127, maxval=255, type=cv2.THRESH_TRUNC)

# ret: is just the cutoff value
plot_image(img_threshold, "Rainbow Image Threshold", cmap_option="gray")

# --------
img2 = cv2.imread("../DATA/crossword.jpg", 0)  # read in grayscale
plot_image(img2, "Crossword Image", cmap_option="gray")

# Binary Threshold
rec2, img2_threshold = cv2.threshold(
    img2, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
plot_image(img2_threshold, "Crossword Image Binary Threshold",
           cmap_option="gray")

# Adaptive Threshold
img2_threshold_adaptive = cv2.adaptiveThreshold(
    img2, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=8)
plot_image(img2_threshold_adaptive,
           "Crossword Image Adaptive Threshold", cmap_option="gray")

# Blend both Thresholds to get the best of both
blended = cv2.addWeighted(src1=img2_threshold, alpha=0.7,
                          src2=img2_threshold_adaptive, beta=0.3, gamma=1)
plot_image(blended,
           "Blended Binary and Adaptive Threshold", cmap_option="gray")

plt.show()
