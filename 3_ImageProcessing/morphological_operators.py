import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image


def load_img():
    blank_img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='ABCDE', org=(50, 300), fontFace=font,
                fontScale=5, color=(255, 255, 255), thickness=25, lineType=cv2.LINE_AA)
    return blank_img


img = load_img()
plot_image(img, "Letters", cmap_option="gray")

# EROSION

kernel = np.ones((5, 5), dtype=np.uint8)
erosion = cv2.erode(img, kernel, iterations=2)
plot_image(erosion, "Letters with Erosion", cmap_option="gray")


# DILATION

# Add noise to use dilatation to remove it
white_noise = np.random.randint(
    low=0, high=2, size=(img.shape[0], img.shape[1]))
white_noise = white_noise*255  # make the image from 0-1 to 0-255
plot_image(white_noise, "White Noise", cmap_option="gray")

noise_img = white_noise+img
plot_image(noise_img, "Letters with Noise", cmap_option="gray")

# use opening to remove the noise
no_noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
plot_image(no_noise_img, "Letters with Noise removed", cmap_option="gray")

black_noise = np.random.randint(
    low=0, high=2, size=(img.shape[0], img.shape[1]))
black_noise = black_noise * (-255)
plot_image(black_noise, "Black Noise", cmap_option="gray")

black_noise_img = img + black_noise
plot_image(black_noise_img, "Letters with Black Noise", cmap_option="gray")

# pass -255 to 0
black_noise_img[black_noise_img == -255] = 0
plot_image(black_noise_img, "Letters with Black Noise2", cmap_option="gray")

# Now we removed the noise in the background and need to now remove the noise in the background:
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
plot_image(no_noise_img, "Letters with Foreground Noise removed",
           cmap_option="gray")


# EDGE DETECTION : GRADIENT
gradient_letters = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plot_image(gradient_letters, "Letters Edge detection from Gradient",
           cmap_option="gray")

# Sobel
img2 = cv2.imread("../DATA/sudoku.jpg", 0)
plot_image(img2, "Sudoku", cmap_option="gray")

# CV_64F to use a 64 float precision
# removes more horizontal lines
sobelx = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)
plot_image(sobelx, "Sudoku Sobel X", cmap_option="gray")

sobely = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)
# removes more vertical lines
plot_image(sobely, "Sudoku Sobel Y", cmap_option="gray")

# Laplacian
laplacian = cv2.Laplacian(img2, cv2.CV_64F)
plot_image(laplacian, "Sudoku Laplacian", cmap_option="gray")


# Blend the outputs
blended = cv2.addWeighted(src1=sobelx, alpha=0.5,
                          src2=sobely, beta=0.5, gamma=0)
plot_image(blended, "Sudoku Blended Sobel with Laplacian", cmap_option="gray")

# Apply a threshold
ret, threshold_img = cv2.threshold(blended, 100, 255, cv2.THRESH_BINARY)
plot_image(threshold_img, "Sudoku Threshold", cmap_option="gray")

kernel = np.ones((4, 4), np.uint8)
gradient2 = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
plot_image(gradient2, "Sudoku Blended Sobel with Laplacian Gradient",
           cmap_option="gray")

plt.show()
