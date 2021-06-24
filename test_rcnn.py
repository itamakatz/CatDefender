import os
import cv2
import torchvision 
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import numpy as np
from PIL import Image, ImageFilter

os.chdir(os.path.dirname(os.path.realpath(__file__)))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_prediction(img_path, threshold):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
    
  """
  img = Image.open(img_path)
  # img = img.filter(ImageFilter.BLUR)
  # img = img.filter(ImageFilter.GaussianBlur(5))
  
  transform = transforms.Compose([transforms.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
  if(len(pred_t) == 0): return None, None
  pred_t = pred_t[-1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  indices = np.argwhere(np.array(pred_class) == 'cat')
  if(len(indices) == 0): return None, None
  indices = np.reshape(indices, indices.size).astype(int)
  pred_boxes = [ pred_boxes[i] for i in indices]
  pred_class = [ pred_class[i] for i in indices]
  return pred_boxes, pred_class


def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3): 
  boxes, pred_cls = get_prediction(img_path, threshold)

  # Get predictions 
  # img = cv2.imread(img_path)
  img = Image.open(img_path)
  # img = img.filter(ImageFilter.BLUR)
  # img = img.filter(ImageFilter.GaussianBlur(5))
  img = np.array(img)
  # Read image with cv2 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  # Convert to RGB 
  plt.figure(figsize=(20,30)) 
  if(boxes is None or pred_cls is None):
    plt.imshow(img) 
  else:
    for i in range(len(boxes)): 
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) 
      # Draw Rectangle with the coordinates 
      cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
      # Write the prediction class 
      # display the output image 
      plt.imshow(img) 
      plt.xticks([]) 
      plt.yticks([]) 
  plt.show()

# object_detection_api('./Images/people.jpg', threshold=0.8)
# object_detection_api('./Images/traffic.jpg', rect_th=2, text_th=1, text_size=1)
# object_detection_api('./Images/cats.jpg', threshold=0.8, rect_th=2, text_th=1, text_size=1)
object_detection_api('./Images/cat2.jpg', threshold=0, rect_th=2, text_th=1, text_size=1)

