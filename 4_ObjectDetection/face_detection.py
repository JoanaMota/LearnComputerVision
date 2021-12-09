import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
from utils import plot_image

nadia = cv2.imread("../DATA/Nadia_Murad.jpg", 0)
denis = cv2.imread("../DATA/Denis_Mukwege.jpg", 0)
solvay = cv2.imread("../DATA/solvay_conference.jpg", 0)
plot_image(nadia, "Nadia", cmap_option="gray")
plot_image(denis, "Denis", cmap_option="gray")
plot_image(solvay, "Solvay", cmap_option="gray")

# FACE CASCADE
face_cascade = cv2.CascadeClassifier(
    "../DATA/haarcascades/haarcascade_frontalface_default.xml")


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h),
                      color=(255, 255, 255), thickness=10)

    return face_img


def detect_face_adjusted(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(
        face_img, scaleFactor=1.17, minNeighbors=3)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 10)

    return face_img


face_nadia = detect_face(nadia)
face_denis = detect_face(denis)
face_solvay = detect_face_adjusted(solvay)
plot_image(face_nadia, "Nadia's Face", cmap_option="gray")
plot_image(face_denis, "Denis's Face", cmap_option="gray")
plot_image(face_solvay, "Solvay's Face", cmap_option="gray")

# EYE CASCADE
eyes_cascade = cv2.CascadeClassifier(
    "../DATA/haarcascades/haarcascade_eye.xml")


def detect_eyes(img):
    eyes_img = img.copy()
    eyes_rects = eyes_cascade.detectMultiScale(
        eyes_img, scaleFactor=1.17, minNeighbors=3)
    for (x, y, w, h) in eyes_rects:
        cv2.rectangle(eyes_img, (x, y), (x+w, y+h), (255, 255, 255), 10)

    return eyes_img


eyes_nadia = detect_eyes(nadia)
eyes_denis = detect_eyes(denis)
plot_image(eyes_nadia, "Nadia's Eyes", cmap_option="gray")
plot_image(eyes_denis, "Denis's Eyes",
           cmap_option="gray")  # Does not work well

# FACE AND EYE DETECTION IN VIDEO
cap = cv2.VideoCapture(2)  # capture another camera, 0 is the default one

while True:
    ret, frame = cap.read()

    # Operations:
    face = detect_face(frame)
    face_and_eyes = detect_eyes(face)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


plt.show()
