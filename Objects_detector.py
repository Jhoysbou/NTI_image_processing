from typing import List, Optional, Union

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

    def __init__(self,
                 rotation_factor=100,
                 width=1280,
                 height=720,
                 circles_pool_length=5,
                 min_area_to_detect=200,
                 circle_factor=1.25,
                 debug_mode=False,
                 min_area_to_compute_mean_colors=20000,
                 daytime="DAY"):

        self._width = width
        self._height = height
        self._pool_size = circles_pool_length
        self._min_area = min_area_to_detect
        self._circle_factor = circle_factor
        self._circles_coords_pool = list()
        self._debug_mode = debug_mode
        self._min_area_mean = min_area_to_compute_mean_colors
        self._daytime = 127 if daytime == "DAY" else 100
        self._rotation_factor = rotation_factor

    # crop image
    def _get_subimage_by_pxs(self, image, start, shift):
        cropped_image = []

        for i in range(shift[1]):
            cropped_image.append([])
            for j in range(shift[0]):
                cropped_image[i].append([])
                cropped_image[i][j] = image[start[1] + i][start[0] + j]

        return np.array(cropped_image)

    def is_rotated(self, frame):
        hsv, thresh = self.__prepare_frame(frame, height=self._height, width=self._width)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < self._min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            area = self._get_subimage_by_pxs(thresh, start=(x, y), shift=(w, h))
            factor = self.__get_difference(area)
            print(factor)
            if factor >= self._rotation_factor:
                return True
            else:
                return False

    def __get_difference(self, frame_thresh):
        reference = cv2.imread('./src/reference.png')
        shape = min(min((frame_thresh.shape, reference.shape)))

        hsv, reference_thresh = self.__prepare_frame(reference, width=shape, height=shape)
        frame_thresh = cv2.resize(frame_thresh, (shape, shape))

        frame_delta = cv2.absdiff(frame_thresh, reference_thresh)
        frame_delta = cv2.dilate(frame_delta, None, iterations=2)
        counter = 0

        for row in frame_delta:
            for pixel in row:
                if pixel == 255:
                    counter += 1
        return counter / shape

    def __prepare_frame(self, frame, width, height):
        frame = cv2.resize(frame, (width, height))

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        blur = cv2.GaussianBlur(hsv, (19, 19), 0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, self._daytime, 255, 1)
        thresh = cv2.bitwise_not(thresh)
        return hsv, thresh

    def _circle_check(self, pool, coords):
        if pool:
            for circles in pool:
                for circle in circles:
                    if max(abs(np.array(circle.get_position()) - np.array(coords))) < \
                            self._circle_factor * circle.get_radius():
                        return False

        return True

    # Color detection by hue
    # check HSV wiki for details

    def _get_color(self, image, x, y):
        area = 0
        for i in range(11):
            if x-i < 0:
                break
            elif x+i >= self._width:
                break
            elif y-i < 0:
                break
            elif y+i >= self._height:
                break
            else:
                area = i

        avg_color = 0

        for i in range(x-area, x+area):
            for j in range(y-area, y+area):
                avg_color += image[j, i, 0]
        avg_color = avg_color / area**2 if area != 0 else image[y, x, 0]
        hsv_color = 255 / 360 * avg_color

        if 0 < hsv_color <= 13 or 330 <= hsv_color:
            return 'RED'
        elif 13 <= hsv_color < 35:
            return 'BLUE'
        elif 35 <= hsv_color < 70:
            return 'ORANGE'
        elif 70 <= hsv_color < 180:
            return 'GREEN'
        elif 180 <= hsv_color < 300:
            return 'YELLOW'
        else:
            return 'no_color'

    def get_objects(self, frame) -> Optional[List[Union[Bucket, Cube]]]:
        if frame is None:
            raise ValueError('Frame is none')

        # Font
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255, 255, 255)

        hsv, thresh = self.__prepare_frame(frame, height=self._height, width=self._width)

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
        circles = cv2.HoughCircles(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, dp=1.5, minDist=100, param1=50, param2=100)
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                circles = list()
                bucket = Bucket(position=(x, y), radius=r, color=self._get_color(hsv, x, y))
                cv2.putText(img=frame, text=str(bucket),
                            org=(x, y),
                            fontFace=font,
                            fontScale=fontScale,
                            color=fontColor)
                circles.append(bucket)
                objects_in_frame.append(bucket)
            self._circles_coords_pool.append(circles)


        # Check all contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self._min_area:
                continue
            elif cv2.contourArea(c) > self._min_area_mean:
                # TODO find dominant color
                pass
                # area = self._get_image_by_px(frame, [x, y], [x + w, y + h])
                # cv2.imshow("cropped", area)
            (x, y, w, h) = cv2.boundingRect(c)
            # area = self._get_image_by_px(frame, [x, y], [x + w, y + h])

            # Validation with circle_check
            if self._circle_check(self._circles_coords_pool, [x, y]):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                cube = Cube(position=(center_x, center_y), color=self._get_color(hsv, x=center_x, y=center_y))
                cv2.putText(img=frame, text=str(cube),
                            org=(x, y),
                            fontFace=font,
                            fontScale=fontScale,
                            color=fontColor)
                objects_in_frame.append(cube)

        cv2.imshow('result', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            return None

        return objects_in_frame
