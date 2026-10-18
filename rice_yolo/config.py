"""Project-relative configuration, independent of the PyCharm working directory."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "yolo官方数据集" / "archive" / "RiceLeafAnnotatedDataset"
CLASSES = [
    "Bacterial_Leaf_Blight", "Brown_Spot", "HealthyLeaf", "Leaf_Blast",
    "Leaf_Scald", "Narrow_Brown_Leaf_Spot", "Neck_Blast", "Rice_Hispa",
]
CLASS_NAMES_ZH = ["白叶枯病", "褐斑病", "健康叶片", "叶瘟病", "叶焦病", "窄褐斑病", "穗颈瘟", "稻铁甲虫危害"]

_DEFAULTS = {
    "source_root": str(SOURCE_ROOT),
    "data_yaml": "datasets/rice/data.yaml",
    "weights_dir": "weights",
    "runs_dir": "runs",
    "train": {
        "model": "yolo11n.pt", "epochs": 50, "imgsz": 640, "batch": 8,
        "workers": 2, "seed": 42, "device": "auto", "patience": 30,
        "optimizer": "auto", "pretrained": True, "amp": True, "cache": False,
    },
    "predict": {"conf": 0.25, "iou": 0.7},
}


def resolve_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def bootstrap_runtime() -> None:
    """Keep library settings and caches inside this project before imports."""
    runtime = PROJECT_ROOT / ".runtime"
    directories = {
        "YOLO_CONFIG_DIR": runtime / "ultralytics",
        "MPLCONFIGDIR": runtime / "matplotlib",
        "TORCH_HOME": runtime / "torch",
    }
    for key, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    # Ultralytics also uses this global switch to enable epoch progress bars.
    os.environ["YOLO_VERBOSE"] = "true"


def load_settings(path: Optional[Union[str, Path]] = None) -> dict:
    """Resolve all project file paths against PROJECT_ROOT, never the caller cwd."""
    config_path = resolve_path(path or "configs/project.yaml")
    if not config_path.is_file():
        raise FileNotFoundError("找不到项目配置：{}".format(config_path))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML。请在 PyCharm 选择已有的 pytorch_gpu 解释器。") from exc
    try:
        supplied = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("无法读取项目配置：{}".format(exc)) from exc
    if not isinstance(supplied, dict):
        raise ValueError("项目配置必须为 YAML 映射。")
    unknown = set(supplied) - set(_DEFAULTS)
    if unknown:
        raise ValueError("未知项目配置项：{}".format(", ".join(sorted(unknown))))
    settings = copy.deepcopy(_DEFAULTS)
    for key, value in supplied.items():
        if key in ("train", "predict"):
            if not isinstance(value, dict):
                raise ValueError("{} 配置必须为映射。".format(key))
            extra = set(value) - set(settings[key])
            if extra:
                raise ValueError("未知 {} 配置项：{}".format(key, ", ".join(sorted(extra))))
            settings[key].update(value)
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} 必须为非空路径。".format(key))
            settings[key] = value
    for key in ("source_root", "data_yaml", "weights_dir", "runs_dir"):
        settings[key] = str(resolve_path(settings[key]))
    return settings

