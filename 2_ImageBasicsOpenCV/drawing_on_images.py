import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.numeric import isclose
from numpy.lib.function_base import blackman
from utils import plot_image

blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)
print(blank_img.shape)
blank_img2 = np.copy(blank_img)

# ---BASIC SHAPES---

# SQUARES
# pt1: top-left cornor, pt2: bottom-right cornor (column, row)
# thickness start on the middle of the line
cv2.rectangle(blank_img, pt1=(384, 50), pt2=(
    500, 150), color=(0, 255, 0), thickness=10)
cv2.rectangle(blank_img, pt1=(200, 200), pt2=(
    300, 300), color=(0, 0, 255), thickness=10)

# CIRCLE
cv2.circle(blank_img, center=(100, 100), radius=50,
           color=(255, 0, 0), thickness=5)
cv2.circle(blank_img, center=(400, 400), radius=50,
           color=(200, 0, 0), thickness=-1)  # -1 fills

# LINES
cv2.line(blank_img, pt1=(100, 100), pt2=(
    400, 400), color=(255, 255, 0), thickness=3)

plot_image(blank_img, "Drawing Basic Shapes")

# ---TEXT---
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img, text="Hello World", org=(10, 500),
            fontFace=font, fontScale=2, color=(255, 0, 255), thickness=3, lineType=(cv2.LINE_AA))
plot_image(blank_img, "Writing Text")

# ---POLYGONS---
vertices = np.array([[100, 300], [200, 200], [400, 300],
                     [200, 400]], dtype=np.int32)
print(vertices)
# Since CV2 requires everything in a certain shape we need to reshape the vertices to have another dimension and pass it as a list
pts = vertices.reshape(-1, 1, 2)
print(pts)
cv2.polylines(blank_img2, [pts], isClosed=True,
              color=(0, 255, 255), thickness=3)
plot_image(blank_img2, "Polygon")

plt.show()
