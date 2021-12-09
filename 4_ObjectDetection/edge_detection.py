import cv2
import numpy as np
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

# Apply Gaussian filter to remove noise
# Find the intensity gradients
# apply non maximum suppression to get rid of spurious response to edge detection
# apply double threshold to determine potential edgers
# filter to getting rid of the weaker edges and only finding the really strong ones

img = cv2.imread("../DATA/sammy_face.jpg")
plot_image(img, "Sammy")

edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plot_image(edges, "Sammy Edges Thresholds 127")

# Lets try to improve the thresholds
# find the median pixel value
med_val = np.median(img)

# Define the Lower threshold as either 0 or 70% of the median value, whichever is greater
lower = int(max(0, 0.7*med_val))
# Define the Upper Threshold to either 130% of the median or the max 255, whichever is smaller
upper = int(min(255, 1.3*med_val)) + 50

edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
plot_image(edges, "Sammy Edges Thresholds " +
           str(lower) + " and " + str(upper))

# Even worse

# Lets improve the image first then
# Blur image
blurred_img = cv2.blur(img, ksize=(5, 5))
edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)
plot_image(edges, "Sammy Blured Edges Thresholds " +
           str(lower) + " and " + str(upper))

# It's much better
# To improve we just need to play with the Threshold values now

plt.show()
