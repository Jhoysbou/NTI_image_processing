import cv2
from Objects_detector import ObjectsDetector



vs = cv2.VideoCapture("./camera/bucket.mov")
# vs = VideoStream(src=0).start()
# time.sleep(0.5)

detector = ObjectsDetector(debug_mode=True)

# Pool for several frames with their circles
circles_coords_pool = list()

while True:
    frame = vs.read()[1]
    objects = detector.get_objects(frame)
    print(objects)

vs.release()
cv2.destroyAllWindows()
