import cv2
import time

cap = cv2.VideoCapture("myVideo.mp4")

if cap.isOpened() == False:
    print("ERROR: File not found or wrong coded used!")

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        # writer wrote at 10 fps
        time.sleep(1/10)  # only for visualization
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:  # video ended
        break

cap.release()
cv2.destroyAllWindows()
