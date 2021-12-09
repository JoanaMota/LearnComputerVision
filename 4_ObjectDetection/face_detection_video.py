import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

# FACE CASCADE
face_cascade = cv2.CascadeClassifier(
    "../DATA/haarcascades/haarcascade_frontalface_default.xml")


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h),
                      color=(0, 255, 255), thickness=3)

    return face_img


def detect_face_adjusted(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(
        face_img, scaleFactor=1.17, minNeighbors=3)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 255), 3)

    return face_img


# EYE CASCADE
eyes_cascade = cv2.CascadeClassifier(
    "../DATA/haarcascades/haarcascade_eye.xml")


def detect_eyes(img):
    eyes_img = img.copy()
    eyes_rects = eyes_cascade.detectMultiScale(
        eyes_img, scaleFactor=1.17, minNeighbors=3)
    for (x, y, w, h) in eyes_rects:
        cv2.rectangle(eyes_img, (x, y), (x+w, y+h), (255, 255, 0), 3)

    return eyes_img


# FACE AND EYE DETECTION IN VIDEO
cap = cv2.VideoCapture(2)  # capture another camera, 0 is the default one

while True:
    ret, frame = cap.read()

    # Operations:
    face = detect_face(frame)
    face_and_eyes = detect_eyes(face)

    cv2.imshow("frame with face and eyes", face_and_eyes)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


plt.show()
