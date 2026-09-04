from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/root/yolo11s-P2-BiFPN.yaml").load(
        "/root/runs/detect/runs/visdrone/yolo11s_p2head_b6/weights/best.pt"
    )
    results = model.train(
        data="/root/autodl-tmp/VisDrone/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=4,
        nbs=64,
        device=0,
        workers=8,
        cache=False,
        project="runs/visdrone",
        name="yolo11s_p2_bifpn_c3k2",
        pretrained=True,
        verbose=True,
    )
    metrics = model.val(split="test")
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
