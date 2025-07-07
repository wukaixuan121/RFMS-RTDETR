import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR()
    model.predict(source='\datasets\images',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  visualize=True
                  )