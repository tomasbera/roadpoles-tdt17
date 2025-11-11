import os
from ultralytics import YOLO




model_path = "yolo11n.pt"


model = YOLO(model_path)

result = model.train(data="../dataset/roadpoles_v1/data.yaml", epochs=5, imgsz=1280)