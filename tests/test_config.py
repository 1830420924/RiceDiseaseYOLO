import os
import tempfile
import unittest
from pathlib import Path

from rice_yolo.config import CLASSES, PROJECT_ROOT, load_settings


class ConfigurationTests(unittest.TestCase):
    def test_settings_paths_do_not_depend_on_pycharm_working_directory(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                settings = load_settings()
            finally:
                os.chdir(old)
        self.assertEqual(Path(settings["data_yaml"]), PROJECT_ROOT / "datasets/rice/data.yaml")
        self.assertEqual(len(CLASSES), 8)
        self.assertEqual(CLASSES[2], "HealthyLeaf")
        self.assertEqual(CLASSES[7], "Rice_Hispa")

    def test_partial_settings_preserve_defaults_without_shared_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("train:\n  batch: 2\n", encoding="utf-8")
            first = load_settings(path)
            self.assertEqual(first["train"]["batch"], 2)
            self.assertEqual(first["train"]["epochs"], 50)
            first["train"]["epochs"] = 1
            self.assertEqual(load_settings(path)["train"]["epochs"], 50)

    def test_unknown_training_setting_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("train:\n  epohcs: 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知"):
                load_settings(path)


if __name__ == "__main__":
    unittest.main()

