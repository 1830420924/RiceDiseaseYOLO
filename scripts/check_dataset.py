"""PyCharm: validate the prepared dataset without importing a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rice_yolo.config import load_settings
from rice_yolo.dataset import DatasetError, print_summary, validate_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="检查八类别水稻检测数据；不训练。")
    parser.add_argument("--data", type=Path, help="准备后数据的 data.yaml")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--verify-images", action="store_true", help="逐张读取和解码图片")
    parser.add_argument("--report", type=Path, help="可选的 JSON 检查报告")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        report = validate_dataset(args.data or Path(settings["data_yaml"]), verify_images=args.verify_images)
        print_summary(report)
        print("数据检查通过，8 类映射及所有集合覆盖完整。")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except (DatasetError, OSError, ValueError) as error:
        print(f"数据检查失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

