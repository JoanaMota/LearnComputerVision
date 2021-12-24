import cv2
import numpy as np
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

# params for ShiTomasi corner detection
corner_track_params = dict(maxCorners=10,
                           qualityLevel=0.3,
                           minDistance=7,
                           blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(10, 10),
                 maxLevel=0,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# maxLevel of 2 represents a resolution of 1/4

cap = cv2.VideoCapture(2)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Points to track
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

mask = np.zeros(prev_frame.shape, np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prevImg=prev_gray, nextImg=frame_gray, prevPts=prev_pts, nextPts=None, **lk_params)

    # Status = 1 if the flow for the corresponding features has been found
    if next_pts is not None:
        good_new = next_pts[status == 1]
        good_prev = prev_pts[status == 1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(mask, (int(x_new), int(y_new)),
                        (int(x_prev), int(y_prev)), (255, 255, 0), 3)
        frame = cv2.circle(frame, (int(x_new), int(y_new)), 8, (0, 0, 255), -1)

    output = cv2.add(frame, mask)
    cv2.imshow("tracking", output)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prev_frame = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)


cap.release()
cv2.destroyAllWindows()
