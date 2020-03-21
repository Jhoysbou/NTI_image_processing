import cv2
import imutils
import numpy as np

vs = cv2.VideoCapture("./camera/WIN_20200321_13_34_37_Pro.mp4")
MIN = 200

h_min = np.array((158, 195, 0), np.uint8)
h_max = np.array((255, 255, 255), np.uint8)

while True:
    frame = vs.read()[1]
    frame = imutils.resize(frame, width=640, height=360)

    if frame is None:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (19, 19), 0)

    thresh = cv2.inRange(blur, h_min, h_max)
    # cv2.imshow("thresh", thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = []

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < MIN:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center.append([int(x + w/2), int(y + h/2)])
    print(center)

    # cv2.imshow("result", frame)
    # key = cv2.waitKey(1) & 0xFF
    #
    # # if the `q` key is pressed, break from the lop
    # if key == ord("q"):
    #     break

vs.release()
cv2.destroyAllWindows()