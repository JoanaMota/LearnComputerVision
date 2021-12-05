import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image, plot_hist


dark_horse = cv2.imread('../DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)
plot_image(show_horse, "Horse")

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
plot_image(show_rainbow, "Rainbow")

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)
plot_image(show_bricks, "Bricks")

# for OpenCV the channels are BGR
hist_bricks = cv2.calcHist([blue_bricks], channels=[0], mask=None,
                           histSize=[256], ranges=[0, 256])  # blue channel
plot_hist(hist_bricks, "Histogram Bricks")

hist_horse = cv2.calcHist([dark_horse], channels=[0], mask=None,
                          histSize=[256], ranges=[0, 256])  # blue channel
plot_hist(hist_horse, "Histogram Horse")

print(rainbow.shape)
# create a mask of the same size as the image
mask = np.zeros(rainbow.shape[:2], np.uint8)
print(mask.shape)
mask[300:400, 100:400] = 255
plot_image(mask, "Mask", cmap_option="gray")

# for the histogram calculation
masked_rainbow = cv2.bitwise_and(rainbow, rainbow, mask=mask)
# for visualization
show_masked_rainbow = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)
plot_image(show_masked_rainbow, "Masked Rainbow")

# calculate histogram
hist_values_red = cv2.calcHist(
    [rainbow], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
plot_hist(hist_values_red, "Histogram Rainbow values Red")
hist_mask_values_red = cv2.calcHist(
    [rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])
plot_hist(hist_mask_values_red, "Histogram Rainbow values Red Masked")


# Histogram equalization
gorilla = cv2.imread("../DATA/gorilla.jpg", 0)
plot_image(gorilla, "Gorilla", "gray")

hist_gorilla = cv2.calcHist(
    [gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plot_hist(hist_gorilla, "Gorilla Histogram")

gorilla_equalized = cv2.equalizeHist(gorilla)
plot_image(gorilla_equalized, "Gorilla Equalized", cmap_option="gray")
hist_gorilla_equalized = cv2.calcHist(
    [gorilla_equalized], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plot_hist(hist_gorilla_equalized, "Gorilla Equalized Histogram")


color_gorilla = cv2.imread("../DATA/gorilla.jpg")
show_color_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
plot_image(show_color_gorilla, "Colored Gorilla")

hsv_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
# equalize the value channel
hsv_gorilla[:, :, 2] = cv2.equalizeHist(hsv_gorilla[:, :, 2])

eq_color_gorilla = cv2.cvtColor(hsv_gorilla, cv2.COLOR_HSV2RGB)
plot_image(eq_color_gorilla, "Equalized color Gorilla")

plt.show()
