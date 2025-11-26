from ultralytics import YOLO
import albumentations as A
from codecarbon import EmissionsTracker

#/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/data.yaml

model_path = "yolo11n.pt"
model = YOLO(model_path)

custom_transforms = [
    A.Blur(blur_limit=7, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.HueSaturationValue(p=0.3),
]

tracker = EmissionsTracker(
    project_name="yolo11s_normal",
    output_dir="./emissions",
    output_file="yolo11s_emission.cvs"

)

tracker.start()
try:
    result = model.train(data="/cluster/home/tomaber/roadpoles-tdt17/yolo/data.yaml", epochs=500, imgsz=1280, augmentations=custom_transforms)

finally:
    tracker.stop()
    val_results = model.val(data="/cluster/home/tomaber/roadpoles-tdt17/yolo/data.yaml", split="val")