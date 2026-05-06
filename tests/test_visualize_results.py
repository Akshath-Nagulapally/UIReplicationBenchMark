import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from PIL import Image
from visualize import scoring
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

    def test_build_results_payload_contains_frontend_friendly_urls_and_scores(self):
        runs_dir = visualize_results.BENCHMARK_DIR / "runs" / f"test-payload-{uuid4().hex}"
        try:
            valid_run = runs_dir / "valid-run"
            valid_target = valid_run / "target.png"
            valid_generated = valid_run / "ai-generated.png"
            self._write_image(valid_target)
            self._write_image(valid_generated)

            with unittest.mock.patch.object(
                visualize_results,
                "SCORE_FUNCTIONS",
                (lambda *_args: 0.25,),
            ):
                with unittest.mock.patch.object(
                    visualize_results,
                    "metric_name",
                    return_value="TEST",
                ):
                    payload = visualize_results.build_results_payload(
                        runs_dir,
                        score_functions=(lambda *_args: 0.25,),
                    )
        finally:
            if runs_dir.exists():
                import shutil

                shutil.rmtree(runs_dir)

        self.assertEqual(payload["scoreNames"], ["TEST"])
        self.assertEqual(payload["results"][0]["name"], "valid-run")
        self.assertEqual(payload["results"][0]["targetImageUrl"], f"/{valid_target.relative_to(visualize_results.BENCHMARK_DIR).as_posix()}")
        self.assertEqual(payload["results"][0]["candidateImageUrl"], f"/{valid_generated.relative_to(visualize_results.BENCHMARK_DIR).as_posix()}")

    def test_build_results_payload_serializes_structured_score_results(self):
        runs_dir = visualize_results.BENCHMARK_DIR / "runs" / f"test-structured-score-{uuid4().hex}"
        try:
            valid_run = runs_dir / "valid-run"
            valid_target = valid_run / "target.png"
            valid_generated = valid_run / "ai-generated.png"
            self._write_image(valid_target)
            self._write_image(valid_generated)

            structured_score = scoring.ScoreResult(
                value=0.75,
                request_success=True,
                reward_hacking=False,
                raw_similarity=0.75,
                reason="Close match.",
                model="openai/gpt-4o",
            )

            with unittest.mock.patch.object(
                visualize_results,
                "metric_name",
                return_value="TEST",
            ):
                payload = visualize_results.build_results_payload(
                    runs_dir,
                    score_functions=(lambda *_args: structured_score,),
                )
        finally:
            if runs_dir.exists():
                import shutil

                shutil.rmtree(runs_dir)

        score_payload = payload["results"][0]["scores"]["TEST"]
        self.assertEqual(score_payload["value"], 0.75)
        self.assertTrue(score_payload["request_success"])
        self.assertFalse(score_payload["reward_hacking"])

    def _write_image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1, 1), (255, 255, 255)).save(path)


if __name__ == "__main__":
    unittest.main()
