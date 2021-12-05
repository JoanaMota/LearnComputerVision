import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.numeric import isclose
from numpy.lib.function_base import blackman
from utils import plot_image

# True while mouse button down, False when mouse button up
drawing = False
ix = -1
iy = -1

# mouse callback function


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(blank_img, center=(x, y), radius=20,
                   color=(0, 255, 0), thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(blank_img, center=(x, y), radius=20,
                   color=(255, 0, 0), thickness=-1)


def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(blank_img, (ix, iy), (x, y),
                          color=(0, 0, 255), thickness=-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(blank_img, (ix, iy), (x, y),
                      color=(0, 0, 255), thickness=-1)


# must have the same name as the plot
cv2.namedWindow("my_drawing")
# cv2.setMouseCallback("my_drawing", draw_circle)
cv2.setMouseCallback("my_drawing", draw_rectangle)

# Show image
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
print(blank_img.shape)

while True:
    cv2.imshow("my_drawing", blank_img)
    if cv2.waitKey(20) & 0xFF == 27:  # 0xFF = 27 : when space key is pressed
        break

cv2.destroyAllWindows()
