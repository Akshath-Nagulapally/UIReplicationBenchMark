import tempfile
import unittest
from pathlib import Path

from PIL import Image
from visualize import visualize_results


class ExperimentDiscoveryTest(unittest.TestCase):
    def test_discover_examples_reads_flat_runs_with_one_generated_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            valid_run = runs_dir / "valid-run"
            valid_target = valid_run / "target.png"
            valid_generated = valid_run / "ai-generated.png"
            self._write_image(valid_target)
            self._write_image(valid_generated)

            self._write_image(runs_dir / "target-only" / "target.png")
            self._write_image(runs_dir / "too-many" / "target.png")
            self._write_image(runs_dir / "too-many" / "ai-generated.png")
            self._write_image(runs_dir / "too-many" / "extra.png")
            (runs_dir / "empty").mkdir()
            (runs_dir / "no-screenshots").mkdir()

            result = visualize_results.discover_examples(runs_dir)

        self.assertEqual(result, [("valid-run", valid_target, valid_generated)])

    def test_discover_examples_supports_legacy_screenshots_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            valid_screenshots = runs_dir / "legacy-run" / "screenshots"
            valid_target = valid_screenshots / "target.png"
            valid_generated = valid_screenshots / "ai-generated.png"
            self._write_image(valid_target)
            self._write_image(valid_generated)

            result = visualize_results.discover_examples(runs_dir)

        self.assertEqual(result, [("legacy-run", valid_target, valid_generated)])

    def _write_image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1, 1), (255, 255, 255)).save(path)


if __name__ == "__main__":
    unittest.main()
