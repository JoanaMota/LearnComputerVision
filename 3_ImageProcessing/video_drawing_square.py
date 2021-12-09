import cv2

cap = cv2.VideoCapture(2)  # capture another camera, 0 is the default one

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # returns float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Top left corner
x = width // 2  # // is used to make sure that the output is integers
y = height // 2

# width and height of rectangle
w = width // 4
h = height // 4

# Bottom left corner x+w, y+h

while True:
    ret, frame = cap.read()

    cv2.rectangle(frame, (x, y), (x+w, y+h),
                  color=(255, 0, 0), thickness=4)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
