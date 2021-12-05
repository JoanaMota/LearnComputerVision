import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image


def load_img():
    img = cv2.imread('../DATA/bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


img = load_img()
plot_image(img, "Bricks Image")


# Gamma Corrections: makes the image brighter or darker
gamma_brighter = 1/4
gamma_darker = 5
# apply the power of gama to every pixel
brighter = np.power(img, gamma_brighter)
plot_image(brighter, "Bricks Image Brighter")
darker = np.power(img, gamma_darker)  # apply the power of gama to every pixel
plot_image(darker, "Bricks Image Darker")

# Low Pass Filter with a 2D convolution
# Add some text first so it is easier to notice the differentiation when applying the filters
font = cv2.FONT_HERSHEY_COMPLEX
img_text = np.copy(img)
cv2.putText(img_text, text="Bricks", org=(
    10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
plot_image(img_text, "Bricks Image with Text")

# Define the kerne to apply the Low Pass Filter (to blur)
kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25
print(kernel)
img_blured = cv2.filter2D(src=img_text, ddepth=-1, kernel=kernel)
# ddepth=-1: means I want the input depth
plot_image(img_blured, "Bricks Image with Text Blured Low Pass Filter")

# CV2 Blur method
img_blured2 = cv2.blur(img_text, ksize=(5, 5))
plot_image(img_blured2, "Bricks Image with Text Blured CV2 Blur")

# Gaussian Blur
img_blured3 = cv2.GaussianBlur(img_text, (5, 5), 10)
plot_image(img_blured3, "Bricks Image with Text Blured Gaussian Blur")

# CV2 Median Blur method
# we keep more the detail with this one
img_blured4 = cv2.medianBlur(img_text, 5)
plot_image(img_blured4, "Bricks Image with Text Blured CV2 media Blur")

# Bilateral Filter
img_blured5 = cv2.bilateralFilter(img_text, 9, 75, 75)
plot_image(img_blured5, "Bricks Image with Text Blured Bilateral Filter")

# Another Image

img2 = cv2.imread("../DATA/sammy.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plot_image(img2, "Sammy")

noise_img2 = cv2.imread("../DATA/sammy_noise.jpg")
plot_image(noise_img2, "Sammy Noise")

median = cv2.medianBlur(noise_img2, 5)
plot_image(noise_img2, "Sammy Noise Reduced")


plt.show()
