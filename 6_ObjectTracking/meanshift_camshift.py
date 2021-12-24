import cv2
import numpy as np

# Face Tracking using MeanShift and CAMShift


def apply_meanshift(prev_track_window):
    # apply meanshift to get the new location
    ret, new_track_window = cv2.meanShift(
        back_proj, prev_track_window, term_criteria)
    x, y, w, h = new_track_window
    img_tracking = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 5)
    return new_track_window, img_tracking


def apply_camshift(prev_track_window):
    # apply meanshift to get the new location
    ret, new_track_window = cv2.CamShift(
        back_proj, prev_track_window, term_criteria)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img_tracking = cv2.polylines(frame, [pts], True, (0, 255, 255), 5)
    return new_track_window, img_tracking


cap = cv2.VideoCapture(2)
ret, init_frame = cap.read()


face_cascade = cv2.CascadeClassifier(
    "../DATA/haarcascades/haarcascade_frontalface_default.xml")

face_rects = face_cascade.detectMultiScale(init_frame)

(face_x, face_y, w, h) = tuple(face_rects[0])  # Fist face that it detects
track_window = (face_x, face_y, w, h)

roi = init_frame[face_y:face_y+h, face_x:face_x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # track_window, img_tracking = apply_meanshift(track_window)
    track_window, img_tracking = apply_camshift(track_window)

    cv2.imshow("tracking", img_tracking)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
