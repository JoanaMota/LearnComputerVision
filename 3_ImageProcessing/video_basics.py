import cv2

cap = cv2.VideoCapture(2)  # capture another camera, 0 is the default one

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # returns float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
print("Width: " + str(width) + " Height: " +
      str(height) + " Frame rate: " + str(frame_rate))

writer = cv2.VideoWriter("myVideo.mp4", cv2.VideoWriter_fourcc(
    *'XVID'), 10, (width, height))  # Fourcc type for Linux

while True:
    ret, frame = cap.read()

    # Operations:
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frame)

    # Save output(write to file)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
