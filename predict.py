"""本地图片或图片文件夹检测；没有训练完成的权重时会明确停止。"""
from __future__ import annotations

import argparse
import json
from multiprocessing import freeze_support

from rice_yolo.config import PROJECT_ROOT, load_settings
from rice_yolo.engine import predict


# PyCharm 直接运行时，将 SOURCE 改成自己的图片或图片文件夹。
CONFIG_PATH = None
SOURCE = str(PROJECT_ROOT / "demo_images")
WEIGHTS_PATH = None
CONFIDENCE = None  # None 沿用配置，默认 0.25。
IOU = None
DEVICE = None
IMAGE_SIZE = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="水稻 8 类图片检测，保存绘框图片、YOLO 标签和逐图 JSON")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--source", default=SOURCE, help="本地图片或图片文件夹，目录会递归读取")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="本地训练后的 best.pt；默认找最近一次训练结果")
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = predict(args.source, load_settings(args.config), weights=args.weights, conf=args.conf, iou=args.iou, device=args.device, imgsz=args.imgsz)
        print("检测完成。未检出目标不能解释为健康。")
        print(json.dumps({key: value for key, value in report.items() if key != "images"}, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"检测失败：{exc}\n请检查图片路径、训练后的 best.pt 和 PyCharm 项目解释器。")
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

