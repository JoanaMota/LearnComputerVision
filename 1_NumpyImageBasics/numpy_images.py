import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from utils import plot_image

pic = Image.open("../DATA/00-puppy.jpg")
# pic.show()
print(type(pic))

pic_arr = np.asarray(pic)
print(type(pic_arr))
print(pic_arr.shape)
plot_image(pic_arr, "Initial Puppy Image")

pic_red = pic_arr[:, :, 0]
plot_image(pic_red, "RED", cmap_option="gray")
pic_green = pic_arr[:, :, 1]
plot_image(pic_green, "GREEN", cmap_option="gray")
pic_blue = pic_arr[:, :, 2]
plot_image(pic_blue, "BLUE", cmap_option="gray")

# delete green channel
pic_no_green = np.copy(pic_arr)
pic_no_green[:, :, 1] = 0
plot_image(pic_no_green, "No Green")
pic_no_green_no_blue = np.copy(pic_no_green)
pic_no_green_no_blue[:, :, 2] = 0
plot_image(pic_no_green_no_blue, "No Green and No Blue")
print(pic_no_green_no_blue.shape)


plt.show()
