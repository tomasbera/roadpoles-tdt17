from ultralytics import YOLO

model = YOLO("../detect/train5/weights/best.pt")

model.predict(
    source="../dataset/roadpoles_v1/test/images",
    project="../test_configs",
    name="./temp_test_save",
    save_txt=True,
    save_conf=True
)