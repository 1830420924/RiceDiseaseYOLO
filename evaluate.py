"""评估本项目训练后的权重，默认使用独立测试集。"""
from __future__ import annotations

import argparse
import json
from multiprocessing import freeze_support

from rice_yolo.config import load_settings
from rice_yolo.engine import evaluate


# PyCharm 直接运行可修改这里。None 自动选择 runs/train 下最新的 best.pt。
CONFIG_PATH = None
WEIGHTS_PATH = None
SPLIT = "test"
DEVICE = None
IMAGE_SIZE = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="水稻 8 类检测模型评估，保存实际指标 JSON 和按类 CSV")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="本地训练完成的 best.pt；默认找最近一次训练结果")
    parser.add_argument("--split", choices=("val", "test"), default=SPLIT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_settings(args.config), weights=args.weights, split=args.split, device=args.device, imgsz=args.imgsz)
        print("评估完成，原始评估图表、metrics.json 和 per_class_metrics.csv 已保存。")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"评估失败：{exc}\n请确认已完成水稻 8 类训练，并检查 README 和 PyCharm 项目解释器。")
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

