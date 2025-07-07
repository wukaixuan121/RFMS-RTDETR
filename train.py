import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR()
    model.train(data='datasets\mydata.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=4,
                workers=4,
                project='runs/train',
                name='exp',
                )