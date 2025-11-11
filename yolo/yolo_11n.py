import os
from ultralytics import YOLO




model_path = "yolo11n.pt"

model = YOLO(model_path)
result = model.train(data="/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml", epochs=2, imgsz=1280)


val_results = model.val(data="/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml", split="val")


print(f"Validation Precision: {val_results.metrics['precision']:.4f}")
print(f"Validation Recall: {val_results.metrics['recall']:.4f}")
print(f"Validation mAP@0.5: {val_results.metrics['map50']:.4f}")
print(f"Validation mAP@0.5:0.95: {val_results.metrics['map50_95']:.4f}")