import os
from ultralytics import YOLO




model_path = "yolo11n.pt"

model = YOLO(model_path)
result = model.train(data="/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml", epochs=100, imgsz=1280)

val_results = model.val(data="/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml", split="val")
