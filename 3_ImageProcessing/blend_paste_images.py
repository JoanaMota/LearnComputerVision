import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_image

img1 = cv2.imread("../DATA/dog_backpack.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("../DATA/watermark_no_copy.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

if img1 is None:
    print("Error loading img1")
    exit(-1)
elif img2 is None:
    print("Error loading img2")
    exit(-1)

print(img1.shape)
print(img2.shape)

plot_image(img1, "Dog Backpack")
plot_image(img2, "Watermark")

# --Blend images of the same size--

# Resize images so they have the same size
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

# Bend images
blended = cv2.addWeighted(src1=img1, alpha=0.7, src2=img2, beta=0.3, gamma=0)
plot_image(blended, "Blended image same size")

# --Overlay a small image on top of a larger image (no blending)--
# Numpy reassignment operation
img1 = cv2.imread("../DATA/dog_backpack.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (600, 600))

large_img = img1
small_img = img2

x_offset = 0
y_offset = 0
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

blended2 = np.copy(large_img)
blended2[y_offset:y_end, x_offset:x_end] = small_img
plot_image(blended2, "Overlay images of different sizes")

# --Blend images of different sizes--
x_offset = large_img.shape[1] - small_img.shape[1]
y_offset = large_img.shape[0] - small_img.shape[0]

roi = large_img[y_offset:large_img.shape[0], x_offset:large_img.shape[0]]
plot_image(roi, "ROI")

roi_blended = cv2.addWeighted(
    src1=roi, alpha=0.5, src2=small_img, beta=0.5, gamma=0)
plot_image(roi_blended, "ROI Blend")

blended3 = np.copy(large_img)
blended3[y_offset:large_img.shape[0],
         x_offset:large_img.shape[0]] = roi_blended
plot_image(blended3, "Blend images of different sizes")

# Now to really blend them we need to apply a mask instead to the small image to remove the background
small_img_gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
# invert image so the parts we want to remove are zeros
mask_inv = cv2.bitwise_not(small_img_gray)
white_background = np.full(small_img.shape, 255, np.uint8)
small_img_no_background = cv2.bitwise_or(
    white_background, white_background, mask=mask_inv)
plot_image(small_img_no_background, "small_img_no_background")
# replace with the original images
small_img_no_background_original = cv2.bitwise_or(
    small_img, small_img, mask=mask_inv)
plot_image(small_img_no_background_original,
           "small_img_no_background_original")

final_roi = cv2.bitwise_or(small_img_no_background_original, roi)
plot_image(final_roi, "Final ROI")

blended4 = np.copy(large_img)
blended4[y_offset:large_img.shape[0],
         x_offset:large_img.shape[0]] = final_roi
plot_image(blended4, "Correctly Blend images of different sizes with")

plt.show()
