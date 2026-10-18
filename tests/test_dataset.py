"""Test conversion and data integrity without importing or running YOLO."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from rice_yolo.config import CLASSES
from rice_yolo.dataset import DatasetError, prepare_dataset, validate_dataset


def snapshot(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "original"
        self.destination = self.base / "prepared"
        self.source.mkdir()
        (self.source / "data.yaml").write_text(yaml.safe_dump({"nc": 8, "names": CLASSES}), encoding="utf-8")
        for split in ("train", "valid", "test"):
            image_dir = self.source / split / "images"
            label_dir = self.source / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir()
            for class_id in range(8):
                filename = f"{split}_{class_id}"
                Image.new("RGB", (32, 32), (class_id*25, 80, 110)).save(image_dir / (filename + ".jpg"))
                (label_dir / (filename + ".txt")).write_text(f"{class_id} 0.5 0.5 0.4 0.6\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_mixed_rows_become_boxes_and_original_bytes_are_preserved(self):
        label = self.source / "train/labels/train_0.txt"
        label.write_text("0 0.1 0.2 0.7 0.2 0.7 0.8 0.1 0.8\n0 0.50 0.50 0.20 0.40\n", encoding="utf-8")
        before = snapshot(self.source)
        result = prepare_dataset(self.source, self.destination)
        rows = (self.destination / "train/labels/train_0.txt").read_text().splitlines()
        polygon_box = list(map(float, rows[0].split()))
        self.assertEqual(polygon_box[0], 0)
        for actual, expected in zip(polygon_box[1:], [0.4, 0.5, 0.6, 0.6]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(rows[1], "0 0.50 0.50 0.20 0.40")
        self.assertEqual(result["counts"]["train"]["boxes"], 9)
        self.assertEqual(before, snapshot(self.source))
        self.assertEqual((self.source / "train/images/train_0.jpg").read_bytes(), (self.destination / "train/images/train_0.jpg").read_bytes())
        self.assertEqual(validate_dataset(self.destination / "data.yaml", verify_images=True)["total_boxes"], 25)

    def test_empty_labels_and_fixed_class_mapping_are_preserved(self):
        Image.new("RGB", (32, 32), "black").save(self.source / "train/images/background.jpg")
        (self.source / "train/labels/background.txt").write_text("\n", encoding="utf-8")
        report = prepare_dataset(self.source, self.destination)
        self.assertEqual(report["counts"]["train"]["empty_labels"], 1)
        self.assertEqual((self.destination / "train/labels/background.txt").read_text(), "")
        document = yaml.safe_load((self.destination / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(document["names"], CLASSES)
        self.assertEqual(document["nc"], 8)
        self.assertEqual(Path(document["path"]), self.destination)
        self.assertEqual(document["val"], "valid/images")

    def test_other_dataset_inside_root_is_never_included(self):
        foreign = self.source / "watermelon/train/labels"
        foreign.mkdir(parents=True)
        (foreign / "bad.txt").write_text("not a rice annotation", encoding="utf-8")
        prepare_dataset(self.source, self.destination)
        self.assertFalse((self.destination / "watermelon").exists())
        self.assertEqual(validate_dataset(self.destination / "data.yaml")["total_images"], 24)

    def test_outside_box_is_rejected_before_creating_destination(self):
        (self.source / "train/labels/train_0.txt").write_text("0 0.05 0.5 0.4 0.6\n", encoding="utf-8")
        before = snapshot(self.source)
        with self.assertRaisesRegex(DatasetError, "超出图片边界"):
            prepare_dataset(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(before, snapshot(self.source))

    def test_invalid_polygon_and_nonfinite_coordinate_are_rejected(self):
        cases = ["0 -0.1 0.2 0.7 0.2 0.7 0.8", "0 0.1 0.2 0.1 0.2 0.1 0.2", "0 nan 0.5 0.4 0.6"]
        for row in cases:
            with self.subTest(row=row):
                (self.source / "train/labels/train_0.txt").write_text(row, encoding="utf-8")
                with self.assertRaises(DatasetError):
                    prepare_dataset(self.source, self.destination)
                self.assertFalse(self.destination.exists())

    def test_repeat_is_idempotent_and_changed_output_requires_explicit_force(self):
        prepare_dataset(self.source, self.destination)
        before = snapshot(self.destination)
        timestamp = (self.destination / "train/images/train_0.jpg").stat().st_mtime_ns
        self.assertTrue(prepare_dataset(self.source, self.destination)["reused"])
        self.assertEqual(before, snapshot(self.destination))
        self.assertEqual(timestamp, (self.destination / "train/images/train_0.jpg").stat().st_mtime_ns)
        changed = self.destination / "train/labels/train_0.txt"
        changed.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "--force"):
            prepare_dataset(self.source, self.destination)
        self.assertEqual(changed.read_text(), "0 0.5 0.5 0.2 0.2")
        report = prepare_dataset(self.source, self.destination, force=True)
        self.assertTrue(Path(report["archive"]).is_dir())
        self.assertEqual((Path(report["archive"]) / "train/labels/train_0.txt").read_text(), "0 0.5 0.5 0.2 0.2")
        self.assertTrue(validate_dataset(self.destination / "data.yaml")["valid"])

    def test_changed_source_never_overwrites_prepared_data_implicitly(self):
        prepare_dataset(self.source, self.destination)
        before = snapshot(self.destination)
        (self.source / "train/labels/train_0.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "--force"):
            prepare_dataset(self.source, self.destination)
        self.assertEqual(before, snapshot(self.destination))

    def test_new_image_label_pair_is_a_change_rather_than_false_reuse(self):
        prepare_dataset(self.source, self.destination)
        Image.new("RGB", (32, 32), "white").save(self.destination / "train/images/new.jpg")
        (self.destination / "train/labels/new.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "--force"):
            prepare_dataset(self.source, self.destination)
        self.assertTrue((self.destination / "train/images/new.jpg").is_file())

    def test_split_alias_cannot_use_training_directory_for_validation(self):
        prepare_dataset(self.source, self.destination)
        data_yaml = self.destination / "data.yaml"
        document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        document["val"] = document["train"]
        data_yaml.write_text(yaml.safe_dump(document), encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "同一个图片目录"):
            validate_dataset(data_yaml)

    def test_wrong_class_mapping_is_rejected(self):
        (self.source / "data.yaml").write_text(yaml.safe_dump({"nc": 8, "names": list(reversed(CLASSES))}), encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "原始顺序"):
            prepare_dataset(self.source, self.destination)

    def test_missing_label_and_missing_class_are_rejected(self):
        prepare_dataset(self.source, self.destination)
        missing = self.destination / "test/labels/test_0.txt"
        missing.unlink()
        with self.assertRaisesRegex(DatasetError, "一一对应"):
            validate_dataset(self.destination / "data.yaml")
        missing.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "未覆盖全部 8 类"):
            validate_dataset(self.destination / "data.yaml")


if __name__ == "__main__":
    unittest.main()

