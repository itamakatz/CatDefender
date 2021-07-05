import threading
import cv2, queue, threading, time
import os
import shutil
import subprocess
import torchvision 
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools
import select, sys
from datetime import datetime

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
cam = VideoCapture(0)

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

class Rectangle():
  def __init__(self, contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    self.x1, self.y1, self.x2, self.y2 = x, y, x+w, y+h

  @staticmethod
  def do_contours_intersect(c1,c2):
    a = Rectangle(c1)
    b = Rectangle(c2)

    x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
    y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
    x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
    y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))

    return x1<x2 and y1<y2

class Running_Time:
  '''Class that stores the initialized time and returns the running time'''

  def __init__(self):
    self.start_time = time.time()

  def get_running_time(self):
    current_time = time.time()
    return f"{((current_time - self.start_time)//60):.0f}:{(current_time - self.start_time)%60:.0f} min:sec"

def check_motion(alpha = 0.96, contour_threshold = 50, cat_threshold=0.1):
  global exitFlag
  print("in check_motion")
  camLock.acquire()
  frame = cam.read()
  camLock.release()

  ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  ref_gray = cv2.GaussianBlur(ref_gray, (21, 21), 0)
  # cv2.imshow('window',frame)

  while not exitFlag:

    if(os.name == "posix"):
      has_input, o, e = select.select( [sys.stdin], [], [], 0.1 ) 
    else:
      has_input = False
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
    countours, heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    countours = [c for c in countours if cv2.contourArea(c) >= contour_threshold]
    import copy
    countours2 = copy.deepcopy(countours)
    countour_indices = list(range(len(countours2)))

    while(True):
      found_intersection = False
      for i in range(len(countour_indices)):
        if(countour_indices[i] < 0): continue
        countour_indices[i] = -0.5
        for j in range(len(countour_indices)):
          if(countour_indices[j] < 0): continue
          if(Rectangle.do_contours_intersect(countours2[i], countours2[j])):
            found_intersection = True
            countours2[i] = np.concatenate([countours2[i], countours2[j]])
            countour_indices[j] = -i-1
            countours2[j] = None

      if(not found_intersection): 
        # countours2 = [pair[0] for pair in list(zip(countours2, countour_indices)) if pair[1] == -0.5]
        countours2 = [c for c in countours2 if c is not None]
        countour_indices = list(range(len(countours2)))
        pairs = []
        for pair in itertools.combinations(list(range(len(countours2))), 2):
          if(countours2[pair[0]] is None or countours2[pair[1]] is None): continue
          if(Rectangle.do_contours_intersect(countours2[pair[0]], countours2[pair[1]])):
            pairs.append(pair)

        if(len(pairs) == 0): break;



    # combined = [] if len(countours) == 0 else np.concatenate(countours)

    # if(len(combined)>0):
    #   (x, y, w, h) = cv2.boundingRect(combined)
    #   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # for i in countours:
    #   (x, y, w, h) = cv2.boundingRect(i)
    #   cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cropped_images = []

    for i in countours2:
      (x, y, w, h) = cv2.boundingRect(i)
      # cropped_images.append(frame[x:x + w,y:y + h,:])
      cropped_images.append(frame[y:y + h,x:x + w,:])
      # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow('window',frame)

    max_pairs = list(zip(countours2, cropped_images))
    max_pairs.sort(reverse=True, key=lambda x: cv2.contourArea(x[0]))
    max_pairs = max_pairs[:1]
    max_pairs = [e for e in max_pairs if cv2.contourArea(e[0]) > 2500]

    for pair in max_pairs:
      timeStr = datetime.now().strftime("%H:%M:%S.%f")
      img_name = f"./predicted_cats/maybe_cat_frame_{timeStr}.png"
      cv2.imwrite(img_name, pair[1])      
      print(f"predicting.. images dim: {pair[1].shape[0]}x{pair[1].shape[1]}")
      running_time = Running_Time()
      cat_detected = get_prediction(pair[1], cat_threshold)
      print(f"finished predicting. running time: {running_time.get_running_time()}")
      if(cat_detected):
        print("detected a cat!")
        result = subprocess.run(
            ["aplay", "-D", "hw:2", "./ding.wav"], capture_output=True, text=True
        )
        print("saving cat frame...")
        timeStr = datetime.now().strftime("%H:%M:%S.%f")
        img_name = f"./predicted_cats/cat_frame_{timeStr}.png"
        cv2.imwrite(img_name, pair[1])
        break

    ref_gray = (ref_gray*alpha + gray2*(1-alpha)).astype(np.uint8)

    if (has_input):
      cli = sys.stdin.readline().strip()
      cli = cli.lower()
      if(cli == 'q'):
        print("Pressed 'q'. Exiting...")
        exitFlag = True
        break  
      if(cli.startswith("alpha")):
        amount = [int(s) for s in cli.split() if s.isdigit()]
        if(len(amount) == 1 and amount[0] > 0 and amount[0] < 1):
          alpha_old = alpha
          alpha = amount[0]
          print(f"changing alpha. current value: {alpha_old}. New value: {alpha}")
      else:
        print(f"received an illegal command: {cli}")

    # k = cv2.waitKey(1)
    # if k%256 == 27:
    #   # ESC pressed
    #   print("Escape hit, closing...")
    #   exitFlag = True
    #   break    
  print("exiting check_motion")

def object_detection_multithreading(threshold=0.5):
  threads = []
  # threads.append(threading.Thread(target=show_cam))
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

# object_detection_multithreading(threshold=0.1)

# check_cat(0.1)
# show_cam()
check_motion()
cam.release()
cv2.destroyAllWindows(alpha = 0.5, contour_threshold = 50, cat_threshold=0.1)