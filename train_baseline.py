"""在 PyCharm 右键 Run 即可训练。仅检查配置时请使用 --dry-run。"""
from __future__ import annotations

import argparse
import json
from multiprocessing import freeze_support

from rice_yolo.config import load_settings
from rice_yolo.engine import train


# PyCharm 直接运行时可修改这里；None 表示沿用 configs/project.yaml。
CONFIG_PATH = None
EXPERIMENT_NAME = "baseline"
EPOCHS = None
BATCH = None
IMAGE_SIZE = None
DEVICE = None  # None 沿用配置；"cpu" 强制 CPU，"0" 使用第 1 张 GPU。
WORKERS = None  # 默认沿用配置中的 2；需要单进程读取时可设为 0。
DRY_RUN = False  # True 只检查数据和参数，绝不加载模型或训练。


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="水稻 8 类 YOLO11n 基线训练（请在 PyCharm 中手动运行）")
    parser.add_argument("--config", default=CONFIG_PATH, help="项目 YAML 配置路径")
    parser.add_argument("--name", default=EXPERIMENT_NAME, help="实验名称，输出目录会自动添加时间及唯一编号")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default=DEVICE, help="auto、cpu 或 GPU 编号，例如 0")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN, help="只校验配置和数据，不加载模型、不下载、不训练")
    parser.add_argument("--debug", action="store_true", help="失败时显示完整异常，便于排查解释器和依赖问题")
    args = parser.parse_args(argv)
    try:
        report = train(
            load_settings(args.config),
            overrides={key: getattr(args, key) for key in ("name", "epochs", "batch", "imgsz", "device", "workers")},
            dry_run=args.dry_run,
        )
        print("配置检查通过，未加载模型或启动训练。" if args.dry_run else "训练已完成。")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"执行失败：{exc}\n请检查 README、数据路径及 PyCharm 项目解释器；需要详细信息时加 --debug。")
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

