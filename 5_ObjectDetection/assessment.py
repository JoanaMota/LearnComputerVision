import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

car = cv2.imread("../DATA/car_plate.jpg")
car_vis = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
plot_image(car_vis, "Car")


def detect_plate(img):
    plate_cascade = cv2.CascadeClassifier(
        "../DATA/haarcascades/haarcascade_russian_plate_number.xml")
    img_copy = img.copy()
    plate_results = plate_cascade.detectMultiScale(
        img_copy, scaleFactor=1.2, minNeighbors=3)
    for x, y, w, h in plate_results:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img_copy


def detect_and_blur_plate(img):
    plate_cascade = cv2.CascadeClassifier(
        "../DATA/haarcascades/haarcascade_russian_plate_number.xml")
    img_copy = img.copy()
    plate_results = plate_cascade.detectMultiScale(
        img_copy, scaleFactor=1.2, minNeighbors=3)
    for x, y, w, h in plate_results:
        roi = img_copy[y:y+h, x:x+w]
        roi_blur = cv2.medianBlur(roi, ksize=25)
        img_copy[y:y+h, x:x+w] = roi_blur

    return img_copy


car_plate = detect_plate(car)
plot_image(car_plate, "Car Plate")
car_plate_blur = detect_and_blur_plate(car)
plot_image(car_plate_blur, "Car Plate Blured")
plt.show()
