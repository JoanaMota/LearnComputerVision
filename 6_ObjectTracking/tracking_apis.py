import cv2
import numpy as np
from matplotlib import cm, image, pyplot as plt


def select_tracker():
    print("Welcome! What Tracker API would you like to use?")
    print("Enter 0 for BOOSTING: ")
    print("Enter 1 for MIL: ")
    print("Enter 2 for KCF: ")
    print("Enter 3 for TLD: ")
    print("Enter 4 for MEDIANFLOW: ")
    choice = input("Please select your tracker: ")

    if choice == '0':
        tracker = cv2.legacy.TrackerBoosting_create()
        tracker_name = "TrackerBoosting"
    if choice == '1':
        tracker = cv2.TrackerMIL_create()
        tracker_name = "TrackerMIL"
    if choice == '2':
        tracker = cv2.TrackerKCF_create()
        tracker_name = "TrackerKCF"
    if choice == '3':
        tracker = cv2.legacy.TrackerTLD_create()
        tracker_name = "TrackerTLD"
    if choice == '4':
        tracker = cv2.legacy.TrackerMedianFlow_create()
        tracker_name = "TrackerMedianFlow"

    return tracker, tracker_name


tracker, tracker_name = select_tracker()
print("Using: " + tracker_name)

cap = cv2.VideoCapture(2)
ret, frame = cap.read()

# Special function allows us to draw on the very first frame our desired ROI
roi = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, roi)

while True:
    ret, frame = cap.read()

    success, roi = tracker.update(frame)

    # roi variable is a tuple of 4 floats
    # We need each value and we need them as integers
    (x, y, w, h) = tuple(map(int, roi))

    # Draw Rectangle as Tracker moves
    if success:
        # Tracking success
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
    else:
        # Tracking failure
        cv2.putText(frame, "Failure to Detect Tracking!!",
                    (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display tracker type on frame
    cv2.putText(frame, tracker_name, (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display result
    cv2.imshow(tracker_name, frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
