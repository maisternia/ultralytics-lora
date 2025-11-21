import ultralytics
print(ultralytics.__version__)
from ultralytics import YOLO
from ultralytics.yolo.engine.trainer import Trainer

model = YOLO("yolov8n.pt")

