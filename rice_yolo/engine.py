"""YOLO 检测流程。导入本模块不会加载模型，也不会启动训练。"""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import CLASSES, PROJECT_ROOT, bootstrap_runtime, load_settings
from .dataset import validate_dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _names_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
        except (KeyError, TypeError) as exc:
            raise ValueError("模型类别编号必须从 0 连续排列。") from exc
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise ValueError("无法读取模型类别名称；请使用本项目训练完成的 best.pt。")


def _check_data(data_yaml: Path, *, verify_images: bool = False) -> dict:
    report = validate_dataset(data_yaml, verify_images=verify_images)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，请在 PyCharm 项目解释器中安装 requirements.txt。") from exc
    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(content, dict) or _names_list(content.get("names")) != list(CLASSES):
        raise ValueError("数据集的 8 类名称或顺序与项目配置不一致，请先运行 prepare_dataset.py。")
    if len(CLASSES) != 8:
        raise ValueError("本项目要求严格的 8 类映射，请检查 rice_yolo/config.py。")
    return report


def _load_yolo(weights: str | Path):
    """唯一的模型构建入口，便于测试确保 dry-run 从不构建模型。"""
    bootstrap_runtime()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            f"缺少训练依赖。请在 PyCharm 选择正确解释器，并安装 {PROJECT_ROOT / 'requirements.txt'}。"
        ) from exc
    try:
        model = YOLO(str(weights), task="detect")
    except Exception as exc:
        raise RuntimeError(
            f"无法加载权重 {weights}。请检查文件完整性和网络；官方初始权重下载地址："
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt。"
            "将文件保存到项目 weights/yolo11n.pt 后重试。"
        ) from exc
    if getattr(model, "task", None) != "detect":
        raise ValueError("权重不是目标检测模型；本项目必须使用 detect 模型。")
    return model


def _check_trained_model(model) -> None:
    if getattr(model, "task", None) != "detect":
        raise ValueError("权重不是 detect 目标检测模型。")
    actual = _names_list(getattr(model, "names", None))
    if actual != list(CLASSES):
        raise ValueError(
            "该权重的类别与水稻 8 类配置不一致。请使用本项目训练后的 best.pt；"
            "weights/yolo11n.pt 是通用初始权重，不能用作已训练的水稻模型。\n"
            f"期望类别：{list(CLASSES)}\n实际类别：{actual}"
        )


def _device(requested: Any) -> Any:
    if requested is not None and str(requested).lower() != "auto":
        return requested
    bootstrap_runtime()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("未安装 PyTorch，请按 README 配置 PyCharm 的训练解释器。") from exc
    return 0 if torch.cuda.is_available() else "cpu"


def _new_output(settings: dict, kind: str, label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = ""
    if label:
        if not re.fullmatch(r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}", label):
            raise ValueError("实验名称仅可使用 1 到 64 个中英文、数字、下划线或短横线。")
        prefix = f"{label}_"
    return Path(settings["runs_dir"]) / kind / f"{prefix}{stamp}_{uuid4().hex[:8]}"


def resolve_weights(settings: dict, weights: str | Path | None = None) -> Path:
    """仅选择本地已训练权重；不会回退到 COCO 初始权重。"""
    if weights is not None:
        candidate = Path(weights).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_file() or candidate.suffix.lower() != ".pt":
            raise ValueError(f"未找到本地 .pt 权重：{candidate}。请通过 --weights 指定训练完成的 best.pt。")
        return candidate
    train_root = Path(settings["runs_dir"]) / "train"
    candidates = list(train_root.glob("*/weights/best.pt")) if train_root.exists() else []
    if not candidates:
        raise ValueError(
            f"尚未找到训练完成的 best.pt：{train_root}。请先在 PyCharm 运行 train_baseline.py，"
            "或用 --weights 指定本地 8 类检测权重。"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime).resolve()


def _training_model(settings: dict, train_args: dict) -> Path:
    value = str(train_args.pop("model", "yolo11n.pt"))
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(settings["weights_dir"]) / candidate
    candidate = candidate.resolve()
    if candidate.suffix.lower() != ".pt":
        raise ValueError("基础训练请使用 .pt 检测初始权重，例如 yolo11n.pt。")
    if not candidate.is_file() and candidate.name != "yolo11n.pt":
        raise ValueError(f"初始权重不存在：{candidate}。请先下载权重或调整 train.model。")
    return candidate


def train(settings: dict | None = None, *, overrides: dict | None = None, dry_run: bool = False) -> dict:
    """实际训练只由入口的显式调用触发；dry_run 不导入 YOLO、不下载、不训练。"""
    settings = settings or load_settings()
    dataset_report = _check_data(Path(settings["data_yaml"]))
    train_args = dict(settings["train"])
    train_args.update({key: value for key, value in (overrides or {}).items() if value is not None})
    experiment_name = train_args.pop("name", None)
    for name in ("epochs", "imgsz", "batch"):
        if not isinstance(train_args.get(name), int) or isinstance(train_args[name], bool) or train_args[name] < 1:
            raise ValueError(f"{name} 必须是正整数。")
    if not isinstance(train_args.get("workers"), int) or train_args["workers"] < 0:
        raise ValueError("workers 必须是大于等于 0 的整数；Windows 建议保持 0。")
    initial_weights = _training_model(settings, train_args)
    output = _new_output(settings, "train", experiment_name)
    args = {
        **train_args,
        "data": str(Path(settings["data_yaml"]).resolve()),
        "task": "detect",
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "val": True,
        "split": "val",
        "save": True,
        "plots": True,
    }
    plan = {
        "dry_run": dry_run,
        "initial_weights": str(initial_weights),
        "output_dir": str(output),
        "classes": list(CLASSES),
        "selection_split": "val",
        "test_used_for_selection": False,
        "training_args": args,
        "dataset": dataset_report,
    }
    if dry_run:
        return plan
    initial_weights.parent.mkdir(parents=True, exist_ok=True)
    args["device"] = _device(args.get("device"))
    model = _load_yolo(initial_weights)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "training_plan.json", plan)
    metrics = model.train(**args)
    _check_trained_model(model)
    best = output / "weights" / "best.pt"
    actual_metrics = getattr(metrics, "results_dict", {})
    result = {
        **plan,
        "dry_run": False,
        "training_args": args,
        "best_weights": str(best) if best.is_file() else None,
        "metrics": _json_safe(actual_metrics) if isinstance(actual_metrics, dict) else {},
    }
    _write_json(output / "run_summary.json", result)
    return result


def _class_names_zh() -> list[str]:
    from . import config

    names = getattr(config, "CLASS_NAMES_ZH", CLASSES)
    return list(names) if len(names) == len(CLASSES) else list(CLASSES)


def evaluate(
    settings: dict | None = None,
    *,
    weights: str | Path | None = None,
    split: str = "test",
    device: Any = None,
    imgsz: int | None = None,
) -> dict:
    settings = settings or load_settings()
    if split not in {"val", "test"}:
        raise ValueError("评估 split 只能是 val 或 test。")
    _check_data(Path(settings["data_yaml"]))
    selected = resolve_weights(settings, weights)
    model = _load_yolo(selected)
    _check_trained_model(model)
    output = _new_output(settings, "evaluate")
    output.mkdir(parents=True, exist_ok=False)
    metrics = model.val(
        data=str(Path(settings["data_yaml"]).resolve()),
        split=split,
        imgsz=imgsz or settings["train"]["imgsz"],
        batch=settings["train"]["batch"],
        workers=settings["train"]["workers"],
        device=_device(device if device is not None else settings["train"].get("device")),
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        task="detect",
        plots=True,
    )
    measured = {int(class_id): index for index, class_id in enumerate(metrics.ap_class_index)}
    class_rows = []
    chinese = _class_names_zh()
    targets = getattr(metrics, "nt_per_class", None)
    for class_id, class_name in enumerate(CLASSES):
        values = metrics.class_result(measured[class_id]) if class_id in measured else (None, None, None, None)
        count = int(targets[class_id]) if targets is not None and len(targets) > class_id else None
        class_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "class_name_zh": chinese[class_id],
            "instances": count,
            "precision": values[0],
            "recall": values[1],
            "mAP50": values[2],
            "mAP50-95": values[3],
            "status": "measured" if class_id in measured else "no_evaluable_targets",
        })
    class_rows = _json_safe(class_rows)
    report = {
        "weights": str(selected),
        "split": split,
        "output_dir": str(output),
        "overall": _json_safe(metrics.results_dict),
        "classes": class_rows,
        "note": "空指标表示该类别没有可评估的目标，未用 0 或其他类别指标填充。",
    }
    _write_json(output / "metrics.json", report)
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    return report


def _image_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path.resolve()]
    if path.is_dir():
        images = sorted(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            return images
    raise ValueError(f"未找到可检测的本地图片：{path}。支持 JPG、PNG、BMP、TIF、WEBP 图片或图片文件夹。")


def _tolist(tensor: Any) -> list:
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)


def predict(
    source: str | Path,
    settings: dict | None = None,
    *,
    weights: str | Path | None = None,
    conf: float | None = None,
    iou: float | None = None,
    device: Any = None,
    imgsz: int | None = None,
) -> dict:
    settings = settings or load_settings()
    images = _image_paths(source)
    selected = resolve_weights(settings, weights)
    confidence = settings["predict"]["conf"] if conf is None else conf
    overlap = settings["predict"]["iou"] if iou is None else iou
    if not 0 <= confidence <= 1 or not 0 <= overlap <= 1:
        raise ValueError("conf 和 iou 必须在 0 到 1 之间。")
    model = _load_yolo(selected)
    _check_trained_model(model)
    output = _new_output(settings, "predict")
    for child in ("images", "labels", "json"):
        (output / child).mkdir(parents=True, exist_ok=False)
    selected_device = _device(device if device is not None else settings["train"].get("device"))
    chinese = _class_names_zh()
    class_counts = {name: 0 for name in CLASSES}
    image_reports = []
    for index, path in enumerate(images, start=1):
        results = model.predict(
            source=str(path),
            task="detect",
            imgsz=imgsz or settings["train"]["imgsz"],
            conf=confidence,
            iou=overlap,
            device=selected_device,
            save=False,
            save_txt=False,
            stream=False,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"单张图片的检测返回数量异常：{path}")
        result = results[0]
        tag = f"{index:06d}_{path.stem}"
        annotated = output / "images" / f"{tag}.jpg"
        labels = output / "labels" / f"{tag}.txt"
        result.save(filename=str(annotated))
        detections = []
        if result.boxes is not None:
            for class_id, score, xyxy in zip(_tolist(result.boxes.cls), _tolist(result.boxes.conf), _tolist(result.boxes.xyxy)):
                class_id = int(class_id)
                if class_id < 0 or class_id >= len(CLASSES):
                    raise ValueError(f"模型返回非法类别编号：{class_id}")
                name = CLASSES[class_id]
                class_counts[name] += 1
                detections.append({
                    "class_id": class_id,
                    "class_name": name,
                    "class_name_zh": chinese[class_id],
                    "confidence": float(score),
                    "bbox_xyxy": [float(value) for value in xyxy],
                })
        if detections:
            result.save_txt(str(labels), save_conf=True)
        else:
            labels.write_text("", encoding="utf-8")
        image_report = {
            "source": str(path),
            "annotated_image": str(annotated),
            "labels": str(labels),
            "status": "detections" if detections else "no_detections",
            "detections": detections,
            "note": "未检出目标不代表健康。" if not detections else "",
        }
        _write_json(output / "json" / f"{tag}.json", image_report)
        image_reports.append(image_report)
    report = {
        "weights": str(selected),
        "output_dir": str(output),
        "image_count": len(images),
        "images_without_detections": sum(not item["detections"] for item in image_reports),
        "detection_count": sum(class_counts.values()),
        "class_counts": class_counts,
        "conf": confidence,
        "iou": overlap,
        "images": image_reports,
        "note": "检测是对图像中的目标定位和分类；未检出目标不能解释为健康。",
    }
    _write_json(output / "summary.json", report)
    return report

