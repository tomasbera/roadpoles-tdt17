import os
from ultralytics import YOLO
import albumentations as A



#/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml
#/cluster/projects/vc/data/mic/closed/MRI_PVS/roadpoles_preprocessed/data.yaml

model_path = "../yolo11s.pt"

model = YOLO(model_path)


custom_transforms = [
    A.Blur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomResizedCrop(p=0.3, scale=(0.5, 0.8), size=(640, 640)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.Normalize()
]


result = model.train(data="/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/data.yaml", epochs=100, imgsz=640, augmentations=custom_transforms)
val_results = model.val(data="/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/data.yaml", split="val")
