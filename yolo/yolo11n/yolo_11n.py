import os
from ultralytics import YOLO


#/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml
#/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml

model_path = "../yolo11n.pt"

model = YOLO(model_path)
result = model.train(data="../dataset/roadpoles_preprocessed/data.yaml", epochs=2, imgsz=640)

val_results = model.val(data="../dataset/roadpoles_preprocessed/data.yaml", split="val")
