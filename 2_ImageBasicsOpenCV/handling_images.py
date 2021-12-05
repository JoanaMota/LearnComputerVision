import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import plot_image

img = cv2.imread("../DATA/00-puppy.jpg")
print(img.shape)

# OpenCV interprets images in BGR format and Matplotlib interprets RGB so we need to convert before plotting
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plot_image(img_rgb, "Initial Puppy Image")

# Read in gray scales
img_gray = cv2.imread("../DATA/00-puppy.jpg", cv2.IMREAD_GRAYSCALE)
plot_image(img_gray, "Initial Puppy Image Gray", cmap_option="gray")
print(img_gray.shape)

# Resize image
img_rgb_resized = cv2.resize(img_rgb, (1000, 400))
plot_image(img_rgb_resized, "RGB Resized")
print(img_rgb_resized.shape)

# Resize image by ratio
img_rgb_resized2 = cv2.resize(img_rgb, (0, 0), img_rgb, 0.5, 0.5)
plot_image(img_rgb_resized2, "RGB Resized by ratio")
print(img_rgb_resized2.shape)
print(img_rgb.shape)

# Flip images
img_rgb_flipped = cv2.flip(img_rgb, 1)  # 0: horizontal axis, 1: vertical axis
plot_image(img_rgb_flipped, "RGB Flipped")

# Save image
# to save return back to BGR
img_bgr_flipped = cv2.cvtColor(img_rgb_flipped, cv2.COLOR_RGB2BGR)
cv2.imwrite("00-puppy-flipped.jpg", img_bgr_flipped)


plt.show()
