# LearnComputerVision
Learn Computer Vision and Deep Learning in Python.

Course: Udemy Python for Computer Vision with OpenCV and Deep Learning Course

## Prerequisites

1. Install [conda](https://www.anaconda.com/products/individual)
2. Setup the environment with all the necessary packages:
    > conda env create -f cvcourse_linux.yml
3. Activate your newly created environment
    > conda activate python-cvcourse

[Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

# Content:
## NumPy and Image Basics
[NumPy Docs](https://numpy.org/doc/stable/user/quickstart.html)

`.arrange()`: creates `int`

`.zeros()`: creates `float`

**`PIL`:** lib for Images.

**[Matplotlib:](https://matplotlib.org/stable/)** for plotting images.

## Image Basics with OpenCV

![](images/coordImage.png)

### Differences between Matplotlib and OpenCV

**Matplotlib** : RGB (RED, GREEN, BLUE)

**OpenCV** : BGR (BLUE, GREEN, RED)

## Image Processing


**Color Mappings**

|         RGB         |         HSL         |         HSV         |
| :-----------------: | :-----------------: | :-----------------: |
| ![](images/rgb.png) | ![](images/hsl.png) | ![](images/hsv.png) |


### [Adding (blending) two images](https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html): 
- `addWeighted` : if images have the same size
- `bitwise_*` : use bitwise functions to combine information of images with different size
  

### [Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)): 
Segment an image into different parts. In case of Binary Threshold it's only 2 parts, white and black. [ThresholdTypes](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)

![](images/thresholding.png)

### [Blurring or Smoothing](https://www.tutorialspoint.com/dip/concept_of_blurring.htm)
- is sometimes combined with edge detection
- Gamma Correction: make image brighter or darker
- [Image Kernels](https://setosa.io/ev/image-kernels/): apply filters to images
- [Gaussian Blurring](https://en.wikipedia.org/wiki/Gaussian_blur)

### [Morphological Operator](https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm)
They are kernels(filters) used to improve the image, like: reducing noise

Operations:
- `morphologyEx` : executes all types of morphological operations
- `erode`
- Gradient : allows us to perform operations such as object detection, object tracking and eventually even image classification. The [Sobel](https://en.wikipedia.org/wiki/Sobel_operator) filter is an Edge Detector and is a very common operator.
  ![](images/gradient.png)

There are a lot of types of [Feature Detection](https://en.wikipedia.org/wiki/Feature_(computer_vision))

### Histogram
Is a visual representation of the distribution of a continuous feature.

**Histogram Equalization** is a method of contrast adjustment based on the image's histogram. It transforms the maximum pixel value into 255 and the minimum to 0.

![](images/hist_equalization.png)





