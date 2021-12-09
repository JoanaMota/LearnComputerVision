from matplotlib import pyplot as plt


def plot_image(image, name, cmap_option=""):
    plt.figure(figsize=(8, 6), dpi=100, num=name)
    if cmap_option == "":
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap_option)


def plot_hist(hist, name):
    plt.figure(figsize=(8, 6), dpi=100, num=name)
    plt.plot(hist)
