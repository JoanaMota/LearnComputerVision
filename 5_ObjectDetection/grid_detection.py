import cv2
import numpy as np
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

# CHESSBOARD
flat_chess = cv2.imread("../DATA/flat_chessboard.png")
plot_image(flat_chess, "Flat Chessboard")

found, corners = cv2.findChessboardCorners(flat_chess, patternSize=(7, 7))
# patternSize is (7,7) because it's not going to be able to detect the outer chess squares because they're missing an edge since they're all the way to the end of the image.

if found:
    cv2.drawChessboardCorners(flat_chess, (7, 7), corners, found)
plot_image(flat_chess, "Flat Chessboard corners ")

# DOT GRID
dots = cv2.imread("../DATA/dot_grid.png")
plot_image(dots, "Dot Grid")

found, corners = cv2.findCirclesGrid(
    dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)

if found:
    cv2.drawChessboardCorners(dots, (10, 10), corners, found)
plot_image(dots, "Dot Grid corners ")

plt.show()
