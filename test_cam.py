# from picamera import PiCamera
# from time import sleep

# camera = PiCamera()

# camera.start_preview()
# sleep(5)
# camera.stop_preview()

# ====

import cv2

cam = cv2.VideoCapture(0)
# width = 2592
# height = 1944
width = 2560
height = 1920
# width = 2304
# height = 1728
# width = 1280
# height = 960
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

# ====

# import cv2
# import pandas as pd
# import cv2

# url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
# table = pd.read_html(url)[0]
# table.columns = table.columns.droplevel()

# cam = cv2.VideoCapture(0)

# for index, row in table[["W", "H"]].iterrows():
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
#     width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     status = cv2.imwrite(f'./test_resolution/res_{row["W"]}x{row["H"]}[WxH].png', frame)
#     print(f'saved: {row["W"]}x{row["H"]}[WxH]')
# cam.release()

# cv2.destroyAllWindows()

# ====

# import pandas as pd
# import cv2

# url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
# table = pd.read_html(url)[0]
# table.columns = table.columns.droplevel()

# cap = cv2.VideoCapture(0)
# resolutions = {}

# for index, row in table[["W", "H"]].iterrows():
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     resolutions[str(width)+"x"+str(height)] = "OK"

# print(resolutions)

# ====

# import cv2, queue, threading, time

# # bufferless VideoCapture
# class VideoCapture:

#   def __init__(self, name):
#     self.cap = cv2.VideoCapture(name)
#     self.q = queue.Queue()
#     t = threading.Thread(target=self._reader)
#     t.daemon = True
#     t.start()

#   # read frames as soon as they are available, keeping only most recent one
#   def _reader(self):
#     while True:
#       ret, frame = self.cap.read()
#       if not ret:
#         break
#       if not self.q.empty():
#         try:
#           self.q.get_nowait()   # discard previous (unprocessed) frame
#         except queue.Empty:
#           pass
#       self.q.put(frame)

#   def read(self):
#     return self.q.get()

# cap = VideoCapture(0)
# while True:
#   time.sleep(.5)   # simulate time between events
#   frame = cap.read()
#   cv2.imshow("frame", frame)
#   if chr(cv2.waitKey(1)&255) == 'q':
#     break