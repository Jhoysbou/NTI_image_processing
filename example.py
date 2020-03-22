import cv2
from Objects_detector import ObjectsDetector
import time


vs = cv2.VideoCapture("./camera/WIN_20200321_13_34_37_Pro.mp4")
# vs = VideoStream(src=0).start()
# time.sleep(0.5)

detector = ObjectsDetector(debug_mode=True)

while True:
    frame = vs.read()[1]
    objects = detector.get_objects(frame)


vs.release()
cv2.destroyAllWindows()
