import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

full_img = cv2.imread("../DATA/sammy.jpg")
full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
plot_image(full_img, "Sammy")

# template image
face = cv2.imread("../DATA/sammy_face.jpg")
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
plot_image(face, "Sammy Face")
height, width, channels = face.shape

# All the 6 methods for comparison in a list
# Note how we are using strings, later on we'll use the eval() function to convert to function
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    # create a copy
    full_copy = full_img.copy()

    method = eval(m)

    # template matching
    res = cv2.matchTemplate(full_copy, face, method)  # res is a heat map

    # Get rectangle
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  # These 2 use the min instead of max
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0]+width, top_left[1]+height)

    cv2.rectangle(full_copy, top_left, bottom_right, (255, 0, 0), 10)

    print(m)

    # Plot the Images
    plt.figure(figsize=(8, 6), dpi=100, num=m)
    plt.subplot(121)
    plt.title("Result of Template Matching")
    plt.imshow(res)

    plt.subplot(122)
    plt.title("Detected Point")
    plt.imshow(full_copy)

plt.show()
