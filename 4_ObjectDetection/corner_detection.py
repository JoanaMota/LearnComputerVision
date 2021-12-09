import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

# ------Harris corner detector------
# FLAT CHESSBOARD
flat_chess = cv2.imread("../DATA/flat_chessboard.png")
flat_chess_rgb = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
plot_image(flat_chess_rgb, "Flat Chessboard")
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
plot_image(gray_flat_chess, "Flat Chessboard Gray", cmap_option="gray")

gray_flat_chess_f32 = np.float32(gray_flat_chess)
corners_gray_flat_chess = cv2.cornerHarris(
    src=gray_flat_chess_f32, blockSize=2, ksize=3, k=0.04)
corners_gray_flat_chess = cv2.dilate(corners_gray_flat_chess, None)

# wherever the result of the corner is greater than one percent of the max value we highlight
limit = 0.01*corners_gray_flat_chess.max()
flat_chess_rgb_corner = np.copy(flat_chess_rgb)
flat_chess_rgb_corner[corners_gray_flat_chess > limit] = [0, 255, 0]
plot_image(flat_chess_rgb_corner, "Flat Chessboard Corners")

# REAL CHESSBOARD
real_chess = cv2.imread("../DATA/real_chessboard.jpg")
real_chess_rgb = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plot_image(real_chess_rgb, "Real Chessboard")
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
plot_image(gray_real_chess, "Real Chessboard Gray", cmap_option="gray")

gray_real_chess_f32 = np.float32(gray_real_chess)
corners_gray_real_chess = cv2.cornerHarris(
    src=gray_real_chess_f32, blockSize=2, ksize=3, k=0.04)
corners_gray_real_chess = cv2.dilate(corners_gray_real_chess, None)

limit = 0.01*corners_gray_real_chess.max()
real_chess_rgb_corners = np.copy(real_chess_rgb)
real_chess_rgb_corners[corners_gray_real_chess > limit] = [0, 255, 0]
plot_image(real_chess_rgb_corners, "Real Chessboard Corners")

# ------Shi-Tomasi Corner Detection & Good Features to Track------
# FLAT CHESSBOARD
good_corners_gray_flat_chess = cv2.goodFeaturesToTrack(
    gray_flat_chess, maxCorners=0, qualityLevel=0.01, minDistance=10)
# use 0 in maxCorners to get all corners instead

# Contrary to Harris Corners this algorithm does not mark the corners
# So we need to:
# convert to ints
good_corners_gray_flat_chess = np.int0(good_corners_gray_flat_chess)
# draw the circles
flat_chess_rgb_good_corner = np.copy(flat_chess_rgb)
for i in good_corners_gray_flat_chess:
    x, y = i.ravel()  # Return a contiguous flattened 1-D array.
    cv2.circle(flat_chess_rgb_good_corner, (x, y), 3, (255, 0, 0), -1)
plot_image(flat_chess_rgb_good_corner, "Flat Chessboard Corners Shi-Tomasi")

# REAL CHESSBOARD
good_corners_gray_real_chess = cv2.goodFeaturesToTrack(
    gray_real_chess, maxCorners=100, qualityLevel=0.01, minDistance=10)

good_corners_gray_real_chess = np.int0(good_corners_gray_real_chess)

real_chess_rgb_good_corner = np.copy(real_chess_rgb)
for i in good_corners_gray_real_chess:
    x, y = i.ravel()  # Return a contiguous realtened 1-D array.
    cv2.circle(real_chess_rgb_good_corner, (x, y), 3, (255, 0, 0), -1)
plot_image(real_chess_rgb_good_corner, "Real Chessboard Corners Shi-Tomasi")

plt.show()
