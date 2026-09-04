from ultralytics import YOLO

model = YOLO(r'yolov8s.pt')
model.predict(source=0, save=True,show=True)