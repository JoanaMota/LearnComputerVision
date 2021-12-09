import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

img = cv2.imread("../DATA/internal_external.png", 0)
plot_image(img, "Internal External Figures", cmap_option="gray")

# The input image has to be a threshold
contours, hierarchy = cv2.findContours(
    img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
# RETR_CCOMP: detects internal and external

# print(len(contours))
# print(hierarchy)
# print(len(hierarchy))

# We need to read the contours according to it's hierarchy to know if they are external or internal
external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)
for i in range(len(contours)):
    # Check, per countour, if hierarchy value is:
    # -1(external)
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)
    # (internal)
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)

# In case of internal contours the hierarchy value can be any value besides -1
# Some of the internal contours have the same hierarchy value because they are inside the same external contour

plot_image(external_contours, "External Contours", cmap_option="gray")
plot_image(internal_contours, "Internal Contours", cmap_option="gray")

plt.show()
