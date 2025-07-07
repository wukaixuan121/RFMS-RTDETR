import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR()
    model.val(data='\datasets\mydata.yaml',
              split='test',
              imgsz=640,
              batch=4,
              project='runs/val',
              name='exp',
              )