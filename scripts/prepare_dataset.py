"""PyCharm: run this file once to prepare the rice detection dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rice_yolo.config import PROJECT_ROOT, load_settings
from rice_yolo.dataset import DatasetError, prepare_dataset, print_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="准备八类别水稻检测数据；不训练、不下载。")
    parser.add_argument("--source", type=Path, help="原始 RiceLeafAnnotatedDataset 根目录")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "datasets" / "rice")
    parser.add_argument("--settings", type=Path, help="可选的项目设置文件")
    parser.add_argument("--force", action="store_true", help="有差异时先归档旧结果，再生成新版")
    parser.add_argument("--report", type=Path, help="可选的 JSON 审计报告输出路径")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        source = args.source or Path(settings["source_root"])
        report = prepare_dataset(source, args.destination, force=args.force)
        print_summary(report)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except (DatasetError, OSError, ValueError) as error:
        print(f"数据准备失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

