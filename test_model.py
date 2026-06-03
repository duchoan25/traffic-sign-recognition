from ultralytics import YOLO

model = YOLO(
    r"C:\Users\admin\runs\detect\train-6\weights\best.pt"
)

results = model.predict(
    source="traffic sign.v8i.yolov8/valid/images",
    conf=0.25,
    save=True
)

print("DONE")