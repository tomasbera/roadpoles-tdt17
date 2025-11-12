import os
from ultralytics import YOLO


#/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml
#/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml

model_path = "../yolo11s.pt"

model = YOLO(model_path)

result = model.train(data="/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/data.yaml", epochs=200, imgsz=1280)
val_results = model.val(data="/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/data.yaml", split="val")
