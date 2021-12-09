import cv2


def draw_rectangle(event, x, y, flags, param):

    global pt1, pt2, topLeft_clicked, botRight_clicked

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if topLeft_clicked == True and botRight_clicked == True:
            topLeft_clicked = False
            botRight_clicked = False
            pt1 = (0, 0)
            pt2 = (0, 0)

        if topLeft_clicked == False:
            pt1 = (x, y)
            topLeft_clicked = True

        elif botRight_clicked == False:
            pt2 = (x, y)
            botRight_clicked = True


# Global variables
pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
botRight_clicked = False

# Connect ot calback function
cap = cv2.VideoCapture(2)  # capture another camera, 0 is the default one
# must have the same name as the plot
cv2.namedWindow("my_drawing_on_video")
cv2.setMouseCallback("my_drawing_on_video", draw_rectangle)


while True:
    ret, frame = cap.read()

    # draw on the frame
    if topLeft_clicked:
        cv2.circle(frame, center=pt1, radius=5,
                   color=(0, 0, 255), thickness=-1)

    # drawing rectangle
    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow("my_drawing_on_video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
