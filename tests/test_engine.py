"""运行流程回归测试：所有模型均为替身，不下载、不训练。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rice_yolo import engine
from rice_yolo.config import CLASSES, load_settings


class FakeResult:
    def __init__(self, detected=True):
        self.boxes = SimpleNamespace(cls=[1], conf=[0.875], xyxy=[[10, 20, 100, 120]]) if detected else None

    def save(self, filename):
        Path(filename).write_bytes(b"annotated image placeholder")

    def save_txt(self, filename, save_conf):
        Path(filename).write_text("1 0.5 0.5 0.2 0.2 0.875\n", encoding="utf-8")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = load_settings()
        for key, value in {"data_yaml": "data.yaml", "weights_dir": "weights", "runs_dir": "runs"}.items():
            self.settings[key] = str(self.root / value)
        self.settings["train"]["device"] = "cpu"
        Path(self.settings["data_yaml"]).write_text(json.dumps({"names": CLASSES}), encoding="utf-8")
        self.weights = self.root / "trained.pt"
        self.weights.write_bytes(b"mock checkpoint")
        self.dataset_patch = patch.object(engine, "validate_dataset", return_value={"ok": True})
        self.dataset_patch.start()
        self.addCleanup(self.dataset_patch.stop)
        self.model = Mock(task="detect")
        self.model.names = dict(enumerate(CLASSES))

    def test_dry_run_does_not_load_download_or_resolve_device(self):
        with patch.object(engine, "_load_yolo") as loader, patch.object(engine, "_device") as device:
            report = engine.train(self.settings, overrides={"epochs": 3, "name": "smoke_test"}, dry_run=True)
        loader.assert_not_called()
        device.assert_not_called()
        self.assertEqual(report["training_args"]["epochs"], 3)
        self.assertEqual(report["selection_split"], "val")
        self.assertFalse(report["test_used_for_selection"])
        self.assertFalse(Path(self.settings["runs_dir"]).exists())
        self.assertTrue(Path(report["output_dir"]).name.startswith("smoke_test_"))

    def test_train_uses_absolute_paths_and_validation_only(self):
        def pretend_train(**kwargs):
            best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
            best.parent.mkdir()
            best.write_bytes(b"mock trained checkpoint")
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.65})

        self.model.train.side_effect = pretend_train
        with patch.object(engine, "_load_yolo", return_value=self.model) as loader:
            report = engine.train(self.settings, overrides={"epochs": 1})
        self.assertTrue(Path(loader.call_args.args[0]).is_absolute())
        args = self.model.train.call_args.kwargs
        self.assertEqual(args["task"], "detect")
        self.assertEqual(args["split"], "val")
        self.assertTrue(args["val"])
        self.assertTrue(Path(args["data"]).is_absolute())
        self.assertEqual(args["workers"], 2)
        self.assertEqual(args["device"], "cpu")
        self.assertTrue(Path(report["best_weights"]).is_file())
        self.assertTrue((Path(report["output_dir"]) / "training_plan.json").is_file())
        self.assertEqual(report["metrics"]["metrics/mAP50(B)"], 0.65)

    def test_each_run_gets_a_unique_directory(self):
        first = engine.train(self.settings, dry_run=True)
        second = engine.train(self.settings, dry_run=True)
        self.assertNotEqual(first["output_dir"], second["output_dir"])

    def test_rejects_wrong_dataset_class_order_before_loading_model(self):
        Path(self.settings["data_yaml"]).write_text(json.dumps({"names": list(reversed(CLASSES))}), encoding="utf-8")
        with patch.object(engine, "_load_yolo") as loader, self.assertRaisesRegex(ValueError, "顺序"):
            engine.train(self.settings, dry_run=True)
        loader.assert_not_called()

    def test_rejects_invalid_training_parameters(self):
        for invalid in (0, -1, "4", True):
            with self.subTest(batch=invalid), self.assertRaises(ValueError):
                engine.train(self.settings, overrides={"batch": invalid}, dry_run=True)

    def test_missing_trained_weights_never_falls_back_to_pretraining(self):
        with self.assertRaisesRegex(ValueError, "尚未找到"):
            engine.resolve_weights(self.settings)
        with self.assertRaisesRegex(ValueError, "本地 .pt"):
            engine.resolve_weights(self.settings, self.root / "missing.pt")

    def test_find_latest_best_only_from_training_directories(self):
        best = Path(self.settings["runs_dir"]) / "train" / "baseline_1" / "weights" / "best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"mock")
        other = Path(self.settings["runs_dir"]) / "predict" / "newer" / "weights" / "best.pt"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"unrelated")
        self.assertEqual(engine.resolve_weights(self.settings), best.resolve())

    def test_rejects_coco_and_wrong_task_checkpoints(self):
        for model in (SimpleNamespace(task="detect", names={0: "person"}), SimpleNamespace(task="classify", names=CLASSES)):
            with self.subTest(task=model.task), self.assertRaises(ValueError):
                engine._check_trained_model(model)

    def test_evaluation_exports_measured_metrics_and_blanks_for_absent_classes(self):
        self.model.val.return_value = SimpleNamespace(
            ap_class_index=[1],
            nt_per_class=[0, 5, 0, 0, 0, 0, 0, 0],
            class_result=lambda index: (0.7, 0.6, 0.8, 0.5),
            results_dict={"metrics/mAP50(B)": 0.8},
        )
        with patch.object(engine, "_load_yolo", return_value=self.model):
            report = engine.evaluate(self.settings, weights=self.weights)
        args = self.model.val.call_args.kwargs
        self.assertEqual(args["split"], "test")
        self.assertEqual(report["classes"][1]["precision"], 0.7)
        self.assertIsNone(report["classes"][0]["precision"])
        self.assertEqual(report["classes"][0]["status"], "no_evaluable_targets")
        self.assertTrue((Path(report["output_dir"]) / "metrics.json").is_file())
        self.assertTrue((Path(report["output_dir"]) / "per_class_metrics.csv").is_file())

    def test_predict_saves_images_labels_json_and_original_names(self):
        source = self.root / "sample.jpg"
        source.write_bytes(b"placeholder; fake model never opens it")
        self.model.predict.return_value = [FakeResult()]
        with patch.object(engine, "_load_yolo", return_value=self.model):
            report = engine.predict(source, self.settings, weights=self.weights)
        item = report["images"][0]
        self.assertEqual(item["detections"][0]["class_name"], CLASSES[1])
        self.assertEqual(report["class_counts"][CLASSES[1]], 1)
        self.assertTrue(Path(item["annotated_image"]).is_file())
        self.assertTrue(Path(item["labels"]).is_file())
        self.assertEqual(len(list((Path(report["output_dir"]) / "json").glob("*.json"))), 1)
        self.assertTrue((Path(report["output_dir"]) / "summary.json").is_file())

    def test_no_detections_is_not_healthy(self):
        source = self.root / "empty.png"
        source.write_bytes(b"mock")
        self.model.predict.return_value = [FakeResult(detected=False)]
        with patch.object(engine, "_load_yolo", return_value=self.model):
            report = engine.predict(source, self.settings, weights=self.weights)
        self.assertEqual(report["images"][0]["status"], "no_detections")
        self.assertIn("不代表健康", report["images"][0]["note"])
        self.assertEqual(report["detection_count"], 0)
        self.assertEqual(Path(report["images"][0]["labels"]).read_text(encoding="utf-8"), "")

    def test_recursive_predict_does_not_overwrite_duplicate_stems(self):
        source = self.root / "pictures"
        for folder in ("a", "b"):
            child = source / folder / "same.jpg"
            child.parent.mkdir(parents=True)
            child.write_bytes(b"mock")
        self.model.predict.return_value = [FakeResult()]
        with patch.object(engine, "_load_yolo", return_value=self.model):
            report = engine.predict(source, self.settings, weights=self.weights)
        self.assertEqual(report["image_count"], 2)
        self.assertEqual(len({item["annotated_image"] for item in report["images"]}), 2)

    def test_prediction_rejects_bad_confidence_before_loading(self):
        source = self.root / "sample.jpg"
        source.write_bytes(b"mock")
        with patch.object(engine, "_load_yolo") as loader, self.assertRaises(ValueError):
            engine.predict(source, self.settings, weights=self.weights, conf=1.1)
        loader.assert_not_called()

    def test_wrong_class_weights_never_reaches_prediction(self):
        source = self.root / "sample.jpg"
        source.write_bytes(b"mock")
        self.model.names = {0: "person"}
        with patch.object(engine, "_load_yolo", return_value=self.model), self.assertRaises(ValueError):
            engine.predict(source, self.settings, weights=self.weights)
        self.model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()

