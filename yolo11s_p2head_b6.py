from ultralytics import YOLO

if __name__ == "__main__":
    # 加载新 yaml 架构，再用 yolo11s.pt 加载能匹配的预训练权重
    model = YOLO("/root/yolo11s-P2-head.yaml").load("yolo11s.pt")

    results = model.train(
        data="/root/autodl-tmp/VisDrone/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=6, nbs=64,
        device=0,
        workers=8,
        cache=False,
        project="runs/visdrone",
        name="yolo11s_p2head_b6",
        pretrained=True,
        verbose=True,
    )

    # 训练完在 test-dev 验证
    metrics = model.val(split="test")
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")