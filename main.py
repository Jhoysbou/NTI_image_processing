import cv2
import imutils
import numpy as np
from objects.Object import Circle, Cube

vs = cv2.VideoCapture("./camera/bucket.mov")

# vs = VideoStream(src=0).start()
# time.sleep(0.5)

"""Constants"""
# Frame resolution
WIDTH = 640
HEIGHT = 360
# Length of pool with previous circles to prevent fluctuations of contours
POOL_LENGTH = 5
# Min area to detect contours
MIN = 200
# Coefficient to prevent fluctuations of contours with circles
DELTA = 1.3


def circle_check(pool, coords):
    if pool:
        for circles in pool:
            for circle in circles:
                if max(abs(np.array(circle.position) - np.array(coords))) < DELTA * circle[2]:
                    return False

    return True


def get_color(image, x, y):
    hsv_color = 255 / 360 * image[y, x, 0]
    if 0 <= hsv_color < 40 or 320 <= hsv_color < 255:
        return 'RED'
    elif 40 <= hsv_color < 70:
        return 'YELLOW'
    elif 70 <= hsv_color < 180:
        return 'GREEN'
    elif 190 <= hsv_color < 280:
        return 'BLUE'


circles_coords_pool = list()

while True:
    frame = cv2.imread('./camera/WIN_20200321_13_37_18_Pro.jpg')
    frame = imutils.resize(frame, width=WIDTH, height=HEIGHT)

    if frame is None:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (19, 19), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
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
            circles.append(Circle(position=(x, y), radius=r, color=get_color(frame, x, y)))
        circles_coords_pool.append(circles)

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < MIN:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if circle_check(circles_coords_pool, [x, y]):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            center.append(Cube(position=(center_x, center_y), color=get_color(frame, x=center_x, y=center_y)))
        print(center)

    cv2.imshow("result", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
