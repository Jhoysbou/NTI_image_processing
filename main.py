import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
import time

# vs = cv2.VideoCapture("./camera/bucket.mov")

vs = VideoStream(src=0).start()
time.sleep(0.5)
POOL_LENGTH = 5
MIN = 200
DELTA = 1.3


def circle_check(pool, coords):
    if pool:
        for circles in pool:
            for circle in circles:
                if max(abs(np.array(circle[:-1]) - np.array(coords))) < DELTA * circle[2]:
                    return False

    return True


circles_coords_pool = list()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640, height=360)

    if frame is None:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (19, 19), 0)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 1)
    thresh = cv2.bitwise_not(thresh)
    cv2.imshow("thresh", thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = []

    if circles_coords_pool is not None and len(circles_coords_pool) >= POOL_LENGTH:
        circles_coords_pool.pop()

    circles = cv2.HoughCircles(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1.5, 20)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            circles = list()
            circles.append([x, y, r])
        circles_coords_pool.append(circles)

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < MIN:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        if circle_check(circles_coords_pool, [x, y]):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center.append([int(x + w / 2), int(y + h / 2)])
        # print(center)

    cv2.imshow("result", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
