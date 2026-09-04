from ultralytics import YOLO

if __name__ == "__main__":
    # 焊缝缺陷实例分割 baseline
    model = YOLO("yolov8s-seg.pt")

    results = model.train(
        data=r"C:/Users/D/Desktop/焊缝缺陷数据集/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,  # RTX 5060 8G 显存，若报 OOM 改成 8
        device=0,
        workers=0,  # Windows 下稳妥，慢但不会出错；想加速可改 4
        cache=False,
        project="runs/weld",
        name="baseline_yolov8s_seg",
        pretrained=True,
        verbose=True,
    )
