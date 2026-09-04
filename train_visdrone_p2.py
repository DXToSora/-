# VisDrone 小目标检测 A/B 训练脚本：baseline(纯 P2 head) vs +HBS 辅助监督
#
# 用法:
#   训练 baseline : python train_visdrone_p2.py
#   训练 +HBS     : python train_visdrone_p2.py --hbs
#
# 常用参数示例:
#   python train_visdrone_p2.py --hbs --imgsz 1280 --batch 4 --epochs 100
#   (小目标检测强烈建议 imgsz>=1280，但 8G 显存需把 batch 调到 2~4)
import argparse

from ultralytics import YOLO

# 本机/服务器通用：省略 path 的 data.yaml（ultralytics 自动用 yaml 所在目录作数据集根）
DATA = "datasets/dataset_visdrone/dataset_visdrone/data_local.yaml"


def main():
    parser = argparse.ArgumentParser(description="VisDrone P2 小目标检测 A/B 训练")
    parser.add_argument("--hbs", action="store_true", help="启用 HBS 辅助监督（训练时 GT mask 平滑背景）")
    parser.add_argument("--data", default=DATA, help="data.yaml 路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="输入分辨率；小目标建议 1280（需大显存）")
    parser.add_argument("--batch", type=int, default=8, help="batch size；8G 显存 1280 分辨率建议 2~4")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0, help="Windows 下 0 最稳，Linux 可设 8")
    parser.add_argument("--cache", action="store_true", help="缓存图片到内存加速（需足够内存）")
    args = parser.parse_args()

    cfg = "yolo11s-P2-HBS.yaml" if args.hbs else "yolo11s-P2-head.yaml"
    name = "yolo11s_p2_hbs" if args.hbs else "yolo11s_p2_baseline"

    # 用 yaml 架构 + yolo11s.pt 预训练权重（只加载能匹配的层，HBS 新层随机初始化）
    model = YOLO(cfg).load("yolo11s.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project="runs/visdrone",
        name=name,
        pretrained=True,
        verbose=True,
    )

    # 训练完在 test-dev 上验证
    metrics = model.val(split="test")
    print(f"\n{name}  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
