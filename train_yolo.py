from ultralytics import YOLO

# Load model gốc YOLOv8
model = YOLO("yolov8n.pt")

# Train với dataset mới
model.train(
    data="traffic sign.v8i.yolov8/data.yaml",
    epochs=40,
    imgsz=416,
    batch=4,
    workers=0
)

print("TRAIN DONE")