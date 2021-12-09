import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

road = cv2.imread("../DATA/road_image.jpg")
plot_image(road, "Road")


def generate_clean_output(input):
    road_copy = input.copy()
    marker = np.zeros(input.shape[:2], dtype=np.int32)
    segments = np.zeros(input.shape, dtype=np.uint8)
    return road_copy, marker, segments


road_copy, marker, segments = generate_clean_output(road)


def create_rgb(i):
    # Use color mapping from matplotlib
    return tuple(np.array(cm.tab10(i)[:3])*255)


# Create colors for the segmentation:
colors = []
for i in range(10):
    colors.append(create_rgb(i))

###
# GLOBAL VARIABLES
n_markers = 10  # 0-9
current_marker = 1
# markers updated by watershed
marks_updated = False

# CALLBACK FUNCTION


def mouse_callback(event, x, y, flags, param):
    global marks_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        # Circles in markers passed to the watershed
        cv2.circle(marker, (x, y), 10, (current_marker), -1)

        # Circles for the user to see
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)

        marks_updated = True

    # if event == cv2.EVENT_


# WHILE LOOP
# must have the same name as the plot
cv2.namedWindow("Road Image")
cv2.setMouseCallback("Road Image", mouse_callback)

# Show image
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
print(blank_img.shape)

while True:
    cv2.imshow("Watershed Segments", segments)
    cv2.imshow("Road Image", road_copy)
    key = cv2.waitKey(1)
    if key == 27:  # when space key is pressed
        break
    # clear all colors
    elif key == ord("c"):  # when c key is pressed
        road_copy, marker, segments = generate_clean_output(road)
    # update color choice
    elif key > 0 and chr(key).isdigit():
        current_marker = int(chr(key))

    # update the markers
    if marks_updated:
        # create a copy because the image is still being manipulated in the window
        marker_copy = marker.copy()
        # Call Watershed on the input image
        cv2.watershed(road, marker_copy)

        segments = np.zeros(road.shape, dtype=np.uint8)
        for color_ind in range(n_markers):
            # coloring the segments
            segments[marker_copy == color_ind] = colors[color_ind]

cv2.destroyAllWindows()


plt.show()
