from typing import List

import cv2
import imutils
import numpy as np

from objects.Object import Bucket, Cube, Object


class ObjectsDetector:
    """
    Constants
    #
    # width Frame resolution
    # height
    #
    # circles_pool_length - Length of pool with previous circles to prevent fluctuations of contours
    # min_area_to_detect - min area to detect contours
    #
    # circle_factor - Coefficient to prevent fluctuations of contours with circles
    # debug_mode - if true call cv2.imshow() to take a look at result

    """
    def __init__(self, width=640,
                 height=360,
                 circles_pool_length=5,
                 min_area_to_detect=200,
                 circle_factor=1.3,
                 debug_mode=False):

        self._width = width
        self._height = height
        self._pool_size = circles_pool_length
        self._min_area = min_area_to_detect
        self._circle_factor = circle_factor
        self._circles_coords_pool = list()
        self._debug_mode = debug_mode

    def _circle_check(self, pool, coords):
        if pool:
            for circles in pool:
                for circle in circles:
                    if max(abs(np.array(circle.get_position()) - np.array(
                            coords))) < self._circle_factor * circle.get_radius():
                        return False

        return True

    # Color detection by hue
    # check HSV wiki for details
    def _get_color(self, image, x, y):
        hsv_color = 255 / 360 * image[y, x, 0]
        if 0 <= hsv_color < 40 or 320 <= hsv_color < 255:
            return 'RED'
        elif 40 <= hsv_color < 70:
            return 'YELLOW'
        elif 70 <= hsv_color < 180:
            return 'GREEN'
        elif 190 <= hsv_color < 280:
            return 'BLUE'

    def get_objects(self, frame) -> List[Cube]:
        frame = imutils.resize(frame, width=self._width, height=self._height)

        if frame is None:
            raise ValueError('frame is none')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv, (19, 19), 0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 100, 255, 1)
        thresh = cv2.bitwise_not(thresh)

        cv2.imshow("thresh", thresh)
        # Store here all
        objects_in_frame = []

        """
        First detect only buckets. It's more simple to detect circle in frame then 
        we iterate each contour and check if it is circle or not.
        We conclude a contour a circle if it has the center nearby a center of already detected circles.
        `get_color` - function that make a verification for contour
        """

        #  Pool must not be bigger than max size
        if self._circles_coords_pool is not None and len(self._circles_coords_pool) >= self._pool_size:
            self._circles_coords_pool.pop()

        # Get circles
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
                circles.append(Bucket(position=(x, y), radius=r, color=self._get_color(frame, x, y)))
            self._circles_coords_pool.append(circles)

        # Check all contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self._min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            # Validation with circle_check
            if self._circle_check(self._circles_coords_pool, [x, y]):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                objects_in_frame.append(
                    Cube(position=(center_x, center_y), color=self._get_color(frame, x=center_x, y=center_y)))
        if self._debug_mode:
            cv2.imshow("result", frame)
            key = cv2.waitKey(1) & 0xFF

        return objects_in_frame
