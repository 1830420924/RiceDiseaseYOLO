"""Validate the rice export and create a separate, pure detection dataset.

This module never imports Ultralytics and never edits the original dataset.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from rice_yolo.config import CLASSES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
MANIFEST_NAME = "preparation_manifest.json"
CONVERTER_VERSION = 1


class DatasetError(ValueError):
    """The dataset is incomplete, inconsistent, or has invalid annotations."""


def _yaml_module():
    try:
        import yaml
        return yaml
    except ImportError as error:
        raise DatasetError("缺少 PyYAML，请按 README 安装 requirements.txt 并选择正确的 PyCharm 解释器。") from error


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_utf8(path: Path, text: str) -> None:
    # Path.write_text(newline=...) is unavailable in the user's Python 3.9.
    # An explicit LF policy also keeps hashes stable on Windows.
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(document: dict[str, Any], path: Path) -> list[str]:
    names = document.get("names")
    if isinstance(names, dict):
        if len(names) != len(CLASSES):
            raise DatasetError(f"类别映射须恰好包含 8 类：{path}")
        try:
            names = [names[i] if i in names else names[str(i)] for i in range(len(CLASSES))]
        except (KeyError, TypeError) as error:
            raise DatasetError(f"类别映射不完整：{path}") from error
    if document.get("nc", len(CLASSES)) != len(CLASSES) or names != CLASSES:
        raise DatasetError(f"必须使用固定的 8 类及原始顺序：{CLASSES}；配置文件：{path}")
    return names


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetError(f"未找到数据配置：{path}。请先运行 scripts/prepare_dataset.py。")
    yaml = _yaml_module()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as error:
        raise DatasetError(f"无法读取数据配置 {path}：{error}") from error
    if not isinstance(document, dict):
        raise DatasetError(f"数据配置必须是 YAML 映射：{path}")
    return document


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DatasetError(f"未找到图片目录：{directory}")
    images = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise DatasetError(f"图片目录为空：{directory}")
    stems = [path.stem.casefold() for path in images]
    duplicate = [stem for stem, count in Counter(stems).items() if count > 1]
    if duplicate:
        raise DatasetError(f"存在同名图片，无法唯一匹配标签：{directory}，示例：{duplicate[:3]}")
    return images


def _parse_label(text: str, path: Path, *, allow_polygons: bool) -> tuple[str, list[int], Counter]:
    rows: list[str] = []
    classes: list[int] = []
    formats: Counter = Counter()
    for number, original in enumerate(text.splitlines(), start=1):
        if not original.strip():
            continue
        fields = original.split()
        location = f"{path} 第 {number} 行"
        if len(fields) != 5 and not (allow_polygons and len(fields) >= 7 and len(fields) % 2 == 1):
            suffix = "检测框须为 5 列；分割多边形须为类别加至少 3 对坐标" if allow_polygons else "检测标签必须为 5 列"
            raise DatasetError(f"{location}：{suffix}。")
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise DatasetError(f"{location}：存在非数值内容。") from error
        if not all(math.isfinite(value) for value in values):
            raise DatasetError(f"{location}：存在 NaN 或无穷大。")
        class_value = values[0]
        if not class_value.is_integer() or not 0 <= class_value < len(CLASSES):
            raise DatasetError(f"{location}：类别 ID 必须为 0–7 的整数。")
        class_id = int(class_value)
        if len(values) == 5:
            x, y, width, height = values[1:]
            output_row = " ".join(fields)
            formats["detection_box"] += 1
        else:
            xs, ys = values[1::2], values[2::2]
            if any(coordinate < 0 or coordinate > 1 for coordinate in values[1:]):
                raise DatasetError(f"{location}：多边形坐标必须在 [0, 1] 内。")
            if len(set(zip(xs, ys))) < 3:
                raise DatasetError(f"{location}：多边形至少需要 3 个不同顶点。")
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            x, y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            output_row = f"{class_id} {x:.12g} {y:.12g} {width:.12g} {height:.12g}"
            formats["segmentation_polygon"] += 1
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise DatasetError(f"{location}：中心坐标须在 [0, 1] 内，宽高须在 (0, 1] 内。")
        tolerance = 1e-9
        if x-width/2 < -tolerance or y-height/2 < -tolerance or x+width/2 > 1+tolerance or y+height/2 > 1+tolerance:
            raise DatasetError(f"{location}：检测框超出图片边界。")
        rows.append(output_row)
        classes.append(class_id)
    return ("\n".join(rows) + "\n" if rows else ""), classes, formats


def _matching_labels(images: list[Path], directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise DatasetError(f"未找到标签目录：{directory}")
    labels = {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"}
    image_stems = {path.stem for path in images}
    missing = image_stems - labels.keys()
    orphan = labels.keys() - image_stems
    if missing or orphan:
        raise DatasetError(f"图片和标签须一一对应：{directory}；缺少标签 {len(missing)} 个，孤立标签 {len(orphan)} 个。")
    return labels


def _read_label(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise DatasetError(f"无法读取标签 {path}：{error}") from error


def _source_scan(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    original_yaml = source_root / "data.yaml"
    _names(_read_yaml(original_yaml), original_yaml)
    records = []
    counts = {}
    source_hashes = {"data.yaml": _file_hash(original_yaml)}
    by_hash = defaultdict(list)
    by_source_name = defaultdict(list)
    for split in SPLITS:
        # Deliberately select the rice splits rather than recursively finding
        # other datasets (the original root also contains a watermelon folder).
        images = _images(source_root / split / "images")
        labels = _matching_labels(images, source_root / split / "labels")
        box_counts = [0] * len(CLASSES)
        image_counts = [0] * len(CLASSES)
        formats = Counter()
        empty = 0
        for image in images:
            label = labels[image.stem]
            converted, ids, original_formats = _parse_label(_read_label(label), label, allow_polygons=True)
            formats.update(original_formats)
            image_hash = _file_hash(image)
            source_hashes[image.relative_to(source_root).as_posix()] = image_hash
            source_hashes[label.relative_to(source_root).as_posix()] = _file_hash(label)
            source_name = re.sub(r"\.rf\.[^.]+$", "", image.stem)
            record = {"split": split, "image": image, "label": label, "label_text": converted, "class_ids": sorted(set(ids)), "image_hash": image_hash}
            records.append(record)
            by_hash[image_hash].append({"split": split, "filename": image.name})
            by_source_name[source_name].append({"split": split, "filename": image.name, "class_ids": sorted(set(ids))})
            empty += not ids
            for class_id in ids:
                box_counts[class_id] += 1
            for class_id in set(ids):
                image_counts[class_id] += 1
        missing_classes = [CLASSES[index] for index, count in enumerate(box_counts) if not count]
        if missing_classes:
            raise DatasetError(f"{split} 集合未覆盖全部 8 类，缺少：{missing_classes}")
        counts[split] = {"images": len(images), "labels": len(labels), "boxes": sum(box_counts), "empty_labels": empty,
                         "class_box_counts": box_counts, "class_image_counts": image_counts, "source_format_counts": dict(formats)}
    collisions = [{"source_name": name, "files": files} for name, files in by_source_name.items() if len({item["split"] for item in files}) > 1]
    differing = sum(len({tuple(item["class_ids"]) for item in group["files"]}) > 1 for group in collisions)
    cross_duplicates = [{"sha256": digest, "files": files} for digest, files in by_hash.items() if len({item["split"] for item in files}) > 1]
    warnings = []
    background_count = sum(value["empty_labels"] for value in counts.values())
    if background_count:
        warnings.append(f"保留 {background_count} 个空标签作为背景样本；未自动补充标注。")
    if collisions:
        if differing == len(collisions):
            warnings.append(f"跨集合源名碰撞 {len(collisions)} 组，均存在类别差异，不能据此证明同源或泄漏；保留原数据划分。")
        else:
            warnings.append(f"跨集合源名碰撞 {len(collisions)} 组，其中 {differing} 组存在类别差异；仅凭文件名不能证明同源或泄漏，保留原数据划分。")
    if cross_duplicates:
        warnings.append(f"发现 {len(cross_duplicates)} 组跨集合字节完全相同的图片；保留原划分，请在正式评估前确认。")
    summary = {"counts": counts, "warnings": warnings, "cross_split_duplicate_byte_groups": cross_duplicates,
               "cross_split_source_name_collisions": collisions, "cross_split_source_name_differing_class_groups": differing}
    return records, summary, source_hashes


def _dataset_yaml(destination: Path) -> str:
    yaml = _yaml_module()
    return yaml.safe_dump({"path": destination.as_posix(), "train": "train/images", "val": "valid/images", "test": "test/images", "nc": len(CLASSES), "names": CLASSES}, allow_unicode=True, sort_keys=False)


def _manifest_matches(destination: Path, source_root: Path, source_hashes: dict[str, str], expected: dict[str, str]) -> bool:
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("converter_version") != CONVERTER_VERSION or manifest.get("source_root") != str(source_root) or manifest.get("source_files") != source_hashes or manifest.get("output_files") != expected:
            return False
        actual_files = {"data.yaml"}
        for split in SPLITS:
            for folder, suffixes in (("images", IMAGE_SUFFIXES), ("labels", {".txt"})):
                directory = destination / split / folder
                if not directory.is_dir():
                    return False
                actual_files.update(path.relative_to(destination).as_posix() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
        if actual_files != expected.keys():
            return False
        for relative, digest in expected.items():
            path = destination / relative
            if not path.is_file() or _file_hash(path) != digest:
                return False
        validate_dataset(destination / "data.yaml")
    except (OSError, ValueError, TypeError):
        return False
    return True


def _remove_staging(staging: Path, parent: Path, prefix: str) -> None:
    resolved = staging.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith(prefix):
        raise DatasetError(f"拒绝清理预期目录以外的路径：{resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def prepare_dataset(source_root: Path, destination: Path, force: bool = False) -> dict[str, Any]:
    """Convert mixed box/polygon labels and copy images without training.

    A changed source or output is never overwritten silently. With ``force``,
    the previous destination is archived after the replacement has validated.
    """
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise DatasetError("原数据与目标目录不得相同或互相包含。")
    if destination.is_symlink():
        raise DatasetError(f"目标目录不能是符号链接：{destination}")
    records, source_summary, source_hashes = _source_scan(source_root)
    yaml_text = _dataset_yaml(destination)
    expected = {"data.yaml": _sha256(yaml_text.encode("utf-8"))}
    for record in records:
        expected[f"{record['split']}/images/{record['image'].name}"] = record["image_hash"]
        expected[f"{record['split']}/labels/{record['label'].name}"] = _sha256(record["label_text"].encode("utf-8"))
    if destination.exists():
        if not destination.is_dir():
            raise DatasetError(f"目标路径不是目录：{destination}")
        if _manifest_matches(destination, source_root, source_hashes, expected):
            report = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))["report"]
            return {**report, "reused": True, "archive": None}
        if not force:
            raise DatasetError(f"{destination} 已存在且与源数据或准备结果不同。未修改现有内容；若需重新准备，请使用 --force，旧目录会先归档。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{destination.name}.preparing-"
    staging = destination.parent / (prefix + uuid4().hex)
    archive = None
    try:
        for split in SPLITS:
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels").mkdir()
        for record in records:
            shutil.copy2(record["image"], staging / record["split"] / "images" / record["image"].name)
            _write_utf8(staging / record["split"] / "labels" / record["label"].name, record["label_text"])
        # Validate the completed staging dataset with a temporary root, then
        # write the final absolute location before publishing it.
        _write_utf8(staging / "data.yaml", _dataset_yaml(staging))
        report = validate_dataset(staging / "data.yaml", verify_images=True)
        _write_utf8(staging / "data.yaml", yaml_text)
        report.update({"source_root": str(source_root), "destination": str(destination), "data_yaml": str(destination / "data.yaml"),
                       "counts": source_summary["counts"], "warnings": source_summary["warnings"], "reused": False,
                       "source_summary": source_summary})
        # Remove any staging path from the saved user-facing report.
        report["root"] = str(destination)
        manifest = {"converter_version": CONVERTER_VERSION, "source_root": str(source_root), "source_files": source_hashes, "output_files": expected, "report": report}
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if destination.exists():
            archive = destination.parent / f"{destination.name}.archive-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
            # Both move targets are checked before moving any directory.
            if destination.parent.resolve() != archive.parent.resolve() or archive.exists():
                raise DatasetError("旧数据归档路径校验失败。")
            destination.rename(archive)
        try:
            staging.rename(destination)
        except Exception:
            if archive is not None and archive.is_dir() and not destination.exists():
                archive.rename(destination)
            raise
        return {**report, "archive": str(archive) if archive else None}
    except Exception:
        _remove_staging(staging, destination.parent, prefix)
        raise


def validate_dataset(data_yaml: Path, verify_images: bool = False) -> dict[str, Any]:
    """Strictly validate a prepared, pure 5-column detection dataset."""
    data_yaml = Path(data_yaml).resolve()
    document = _read_yaml(data_yaml)
    _names(document, data_yaml)
    root = Path(document.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()
    counts = {}
    class_counts = {}
    class_image_counts = {}
    warnings = []
    source_names = defaultdict(list)
    split_directories = {}
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        entry = document.get(key)
        if not isinstance(entry, str) or not entry:
            raise DatasetError(f"配置缺少 {key} 图片目录：{data_yaml}")
        image_dir = Path(entry)
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        image_dir = image_dir.resolve()
        if image_dir.name != "images":
            raise DatasetError(f"图片目录须命名为 images，以对应同级 labels：{image_dir}")
        if image_dir in split_directories:
            raise DatasetError(f"{split} 和 {split_directories[image_dir]} 指向同一个图片目录，训练/验证/测试集合必须分别配置：{image_dir}")
        split_directories[image_dir] = split
        images = _images(image_dir)
        labels = _matching_labels(images, image_dir.parent / "labels")
        box_counts = [0] * len(CLASSES)
        image_counts = [0] * len(CLASSES)
        empty = 0
        for image in images:
            _, ids, _ = _parse_label(_read_label(labels[image.stem]), labels[image.stem], allow_polygons=False)
            empty += not ids
            for class_id in ids:
                box_counts[class_id] += 1
            for class_id in set(ids):
                image_counts[class_id] += 1
            source_name = re.sub(r"\.rf\.[^.]+$", "", image.stem)
            source_names[source_name].append({"split": split, "class_ids": tuple(sorted(set(ids)))})
            if verify_images:
                try:
                    from PIL import Image
                    with Image.open(image) as opened:
                        opened.verify()
                    with Image.open(image) as opened:
                        opened.load()
                except ImportError as error:
                    raise DatasetError("读取图片验证需要 Pillow，请先安装 requirements.txt。") from error
                except (OSError, ValueError) as error:
                    raise DatasetError(f"图片无法读取：{image}；{error}") from error
        absent = [CLASSES[index] for index, count in enumerate(box_counts) if count == 0]
        if absent:
            raise DatasetError(f"{split} 集合未覆盖全部 8 类，缺少：{absent}")
        counts[split] = {"images": len(images), "labels": len(labels), "boxes": sum(box_counts), "empty_labels": empty}
        class_counts[split], class_image_counts[split] = box_counts, image_counts
    empty_total = sum(item["empty_labels"] for item in counts.values())
    if empty_total:
        warnings.append(f"保留 {empty_total} 个空标签作为背景样本；未自动补充标注。")
    collisions = [files for files in source_names.values() if len({item["split"] for item in files}) > 1]
    differing = sum(len({item["class_ids"] for item in files}) > 1 for files in collisions)
    if collisions:
        difference_text = "均存在类别差异" if differing == len(collisions) else f"其中 {differing} 组存在类别差异"
        warnings.append(f"跨集合源名碰撞 {len(collisions)} 组，{difference_text}，不能据此证明同源或泄漏；保留原数据划分。")
    return {"valid": True, "data_yaml": str(data_yaml), "root": str(root), "names": list(CLASSES), "counts": counts,
            "class_counts": class_counts, "class_image_counts": class_image_counts,
            "total_images": sum(item["images"] for item in counts.values()), "total_boxes": sum(item["boxes"] for item in counts.values()),
            "images_verified": verify_images, "warnings": warnings,
            "cross_split_source_name_collision_groups": len(collisions), "cross_split_source_name_differing_class_groups": differing}


def print_summary(report: dict[str, Any]) -> None:
    print(f"数据配置：{report['data_yaml']}")
    for split, counts in report["counts"].items():
        print(f"{split}: {counts['images']} 张图片，{counts['boxes']} 个目标，{counts['empty_labels']} 个空标签")
    for warning in report.get("warnings", []):
        print(f"提示：{warning}")
    if report.get("reused"):
        print("源文件与准备结果一致，已复用现有数据。")
    if report.get("archive"):
        print(f"原准备结果已归档：{report['archive']}")

