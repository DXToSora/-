import cv2

from ultralytics import YOLO

model = YOLO(r"yolov8s.pt")
results = model.predict(source=0, show=True)

for result in results:
    plotted = result.plot()
    cv2.imshow("YOLOv8 Inference", plotted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
