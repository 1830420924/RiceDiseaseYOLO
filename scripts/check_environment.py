"""只检查解释器和已安装依赖；绝不安装、更新、构建模型或训练。"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rice_yolo.config import bootstrap_runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="检查当前 PyCharm/Python 环境，不安装依赖、不加载模型、不训练")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "environment.json")
    args = parser.parse_args(argv)
    bootstrap_runtime()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "packages": {},
        "cuda": {},
        "issues": [],
        "training_started": False,
        "model_constructed": False,
    }
    modules = {"torch": "torch", "torchvision": "torchvision", "ultralytics": "ultralytics", "PyYAML": "yaml", "Pillow": "PIL", "numpy": "numpy", "opencv-python": "cv2", "matplotlib": "matplotlib"}
    for distribution, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = str(getattr(module, "__version__", "unknown"))
            report["packages"][distribution] = {"version": version, "import_ok": True}
        except Exception as exc:
            report["packages"][distribution] = {"import_ok": False, "error": str(exc)}
            report["issues"].append(f"{distribution} 无法导入：{exc}")
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            available = bool(torch.cuda.is_available())
            report["cuda"] = {
                "available": available,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count() if available else 0,
                "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if available else [],
                "automatic_training_device": "0" if available else "cpu",
            }
        except Exception as exc:
            report["issues"].append(f"GPU 环境检查失败：{exc}")
    report["ready"] = not report["issues"]
    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"环境报告：{output}")
    if report["issues"]:
        print("请在 PyCharm 选择预期解释器，并按 README 安装 requirements.txt；本脚本不会替您安装或升级。")
    else:
        print("依赖检查通过。此检查没有构建模型或启动训练。")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

