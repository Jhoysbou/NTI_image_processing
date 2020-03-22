import cv2
from Objects_detector import ObjectsDetector
import time
from imutils.video import VideoStream

vs = cv2.VideoCapture("./camera/bucket.mov")
# vs = VideoStream(src=0).start()
# time.sleep(0.5)

detector = ObjectsDetector(debug_mode=True)

while True:
    frame = vs.read()[1]
    objects = detector.get_objects(frame)



vs.release()
cv2.destroyAllWindows()
