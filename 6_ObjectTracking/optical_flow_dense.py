import cv2
import numpy as np
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

cap = cv2.VideoCapture(2)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(prev_frame)
hsv_mask[:, :, 1] = 255  # Fully saturated

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # flow : direction vector
    # contains magnitude and angle information
    # we need to convert it to polar coordinates

    mag, ang = cv2.cartToPolar(
        flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)

    hsv_mask[:, :, 0] = ang/2
    hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("tracking", bgr)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prev_gray = frame_gray


cap.release()
cv2.destroyAllWindows()
