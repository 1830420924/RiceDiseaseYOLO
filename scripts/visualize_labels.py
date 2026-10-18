"""Create an eight-class ground-truth preview; this does not run inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from PIL import Image, ImageDraw, ImageFont

from rice_yolo.config import CLASSES, PROJECT_ROOT, load_settings
from rice_yolo.dataset import DatasetError, IMAGE_SUFFIXES, validate_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成 8 类真实标签拼图，不运行模型。")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "label_preview.jpg")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        data_yaml = (args.data or Path(settings["data_yaml"])).resolve()
        report = validate_dataset(data_yaml)
        document = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
        root = Path(report["root"])
        image_dir = Path(document["train"])
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        label_dir = image_dir.parent / "labels"
        examples = {}
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows = [list(map(float, row.split())) for row in (label_dir / (image.stem + ".txt")).read_text(encoding="utf-8-sig").splitlines() if row.strip()]
            for row in rows:
                examples.setdefault(int(row[0]), (image, rows))
            if len(examples) == len(CLASSES):
                break
        cell_width, cell_height = 380, 430
        canvas = Image.new("RGB", (4 * cell_width, 2 * cell_height), "#12202f")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font = small = ImageFont.load_default()
        colors = ["#59d997", "#ffbf60", "#60d4ff", "#f681a0", "#b99aff", "#ffec77", "#f58c65", "#82e7da"]
        for class_id, name in enumerate(CLASSES):
            image_path, rows = examples[class_id]
            with Image.open(image_path) as original:
                image = original.convert("RGB")
            image.thumbnail((cell_width - 20, cell_width - 20))
            left = (class_id % 4) * cell_width + (cell_width - image.width) // 2
            top = (class_id // 4) * cell_height + 10
            canvas.paste(image, (left, top))
            for cls, x, y, width, height in rows:
                color = colors[int(cls)]
                box = (left + (x-width/2)*image.width, top + (y-height/2)*image.height,
                       left + (x+width/2)*image.width, top + (y+height/2)*image.height)
                draw.rectangle(box, outline=color, width=3)
                draw.text((box[0]+3, box[1]+2), str(int(cls)), fill=color, font=font, stroke_width=1, stroke_fill="black")
            title_top = (class_id // 4) * cell_height + 376
            draw.text(((class_id % 4)*cell_width+10, title_top), f"{class_id}  {name}", fill=colors[class_id], font=font)
            short_name = image_path.name[:47]
            draw.text(((class_id % 4)*cell_width+10, title_top+27), short_name, fill="#c1cbd6", font=small)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.output, quality=94)
        print(f"真实标签预览已保存：{args.output.resolve()}")
        return 0
    except (DatasetError, OSError, ValueError, KeyError) as error:
        print(f"标签预览生成失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

