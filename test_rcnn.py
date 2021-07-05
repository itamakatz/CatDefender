import threading
import cv2, queue, threading, time
import os
import subprocess
import torchvision 
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, signal

class VideoCapture:
  '''bufferless VideoCapture'''

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    width = 2560 # 2592
    height = 1920 # 1944
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    self.q = queue.Queue()
    self.stop_thread = False
    self.t = threading.Thread(target=self._reader)
    self.t.daemon = True
    self.t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      if(self.stop_thread): return
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put_nowait(frame)

  def read(self):
    return self.q.get()

  def release(self):
    self.stop_thread = True
    while(self.t.is_alive()):
      print('VideoCapture thread still running')
      time.sleep(.5)
    print('VideoCapture thread released')
    self.cap.release()

def memoize(f):
    def wrapped(*args, **kwargs):
        if hasattr(wrapped, '_cached_val'):
            return wrapped._cached_val
        result = f(*args, **kwargs)
        wrapped._cached_val = result
        return result
    return wrapped

@memoize
def lazy_model():
    print("Creating model once which is expensive...")
    time.sleep(1)

    # CACHE_PATH = "/home/pi/.cache/torch/hub/checkpoints"
    # CACHE_BACKUP_PATH = "/home/pi/.cache/torch/hub/checkpoints_backup"
    # if(os.path.exists(CACHE_PATH)):
    #   shutil.rmtree(CACHE_PATH)
    # if(os.path.exists(CACHE_BACKUP_PATH)):
    #   shutil.copytree(CACHE_BACKUP_PATH, CACHE_PATH)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True) 
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True) 
    model.eval()  
    return model

os.chdir(os.path.dirname(os.path.realpath(__file__)))

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

exitFlag = False
camLock = threading.Lock()
countoursLock = threading.Lock()
cam = VideoCapture(0)
countours = []

def get_prediction(img, threshold):

  transform = transforms.Compose([transforms.ToTensor()])
  img = transform(img)
  model = lazy_model()
  pred = model([img])
  pred_class = list(pred[0]['labels'].detach().numpy())
  pred_score = list(pred[0]['scores'].detach().numpy())
  class_scores_pair = list(zip(pred_class, pred_score))
  pred_t = [pair for pair in class_scores_pair if COCO_INSTANCE_CATEGORY_NAMES[pair[0]]=='cat' and pair[1]>threshold]
  return len(pred_t) > 0 

def show_cam():
  print("in show_cam")
  global exitFlag 
  cv2.namedWindow("cam")
  while not exitFlag:

    camLock.acquire()
    frame = cam.read()
    camLock.release()

    if frame is None:
      print("failed to grab frame")
      exitFlag = True
      break
    cv2.imshow("cam", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
      # ESC pressed
      print("Escape hit, closing...")
      exitFlag = True
      break
  print("exiting show_cam")

def check_cat(threshold = 0.1):
  print("in check_detect")
  while not exitFlag:

    camLock.acquire()
    frame = cam.read()
    camLock.release()

    print("sending image to model")
    cat_found = get_prediction(frame, threshold)
    if(cat_found):
      print("detected a cat!")
      result = subprocess.run(
          ["aplay", "-D", "hw:2", "./ding.wav"], capture_output=True, text=True
      )
  print("exiting check_detect")

def check_motion(alpha = 0.96):
  global exitFlag, countours
  print("in check_motion")
  camLock.acquire()
  frame = cam.read()
  camLock.release()

  ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  ref_gray = cv2.GaussianBlur(ref_gray, (21, 21), 0)
  cv2.imshow('window',frame)

  while not exitFlag:
    camLock.acquire()
    frame = cam.read()
    camLock.release()
    
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    deltaframe=cv2.absdiff(ref_gray,gray2)
    # cv2.imshow('delta',deltaframe)
    threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None)
    # cv2.imshow('threshold',threshold)
    local_countours, heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    local_countours = [c for c in local_countours if cv2.contourArea(c) >= 50]

    for i in local_countours:
      (x, y, w, h) = cv2.boundingRect(i)
      cropped_im = frame[x:x + w,y:y + h,:]
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('window',frame)

    ref_gray = (ref_gray*alpha + gray2*(1-alpha)).astype(np.uint8)

    k = cv2.waitKey(1)
    if k%256 == 27:
      # ESC pressed
      print("Escape hit, closing...")
      exitFlag = True
      break    
  print("exiting check_motion")

def object_detection_multithreading(threshold=0.5):
  threads = []
  threads.append(threading.Thread(target=show_cam))
  # threads.append(threading.Thread(target=check_cat))
  threads.append(threading.Thread(target=check_motion))
  for t in threads:
   t.start()
  for t in threads:
   t.join()   
  # camLock.release()
 
# img = Image.open('./Images/cat3.jpeg')
# plt.figure(figsize=(20,30)) 
# plt.imshow(img)   
# plt.show() 
# get_prediction(img, 0.1)

object_detection_multithreading(threshold=0.1)

# check_cat(0.1)
# show_cam()
# check_motion()
cam.release()
cv2.destroyAllWindows()