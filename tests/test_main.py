import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import image_sources
import numpy as np
import main
from PIL import Image
from visualize import scoring


class ImageComparisonTest(unittest.TestCase):
    def test_get_imitation_image_mounts_unique_run_output_directory(self):
        class FakeUUID:
            hex = "test-run-id"

        run_dir = Path(__file__).resolve().parents[1] / "runs" / FakeUUID.hex
        shutil.rmtree(run_dir, ignore_errors=True)

        try:
            with patch.object(main, "uuid4", return_value=FakeUUID()):
                with patch.object(main, "_wait_for_no_running_image_containers"):
                    with patch.object(main, "_validate_run_outputs"):
                        with patch.object(main.subprocess, "run") as run:
                            result = main.get_imitation_image("copy the target UI", "https://example.com")

            self.assertEqual(result, run_dir)
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(run.call_count, 2)

            docker_run_command = run.call_args_list[1].args[0]
            self.assertIn("--name", docker_run_command)
            self.assertIn(f"{main.IMAGE_NAME}-{FakeUUID.hex}", docker_run_command)
            self.assertIn("-v", docker_run_command)
            self.assertIn(f"{run_dir}:{main.CONTAINER_OUTPUT_DIR}", docker_run_command)
            self.assertIn("UI_REPLICATE_PROMPT=copy the target UI", docker_run_command)
            self.assertIn("UI_REPLICATE_TARGET_IMAGE_URL=https://example.com", docker_run_command)
            self.assertNotIn("-p", docker_run_command)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_get_imitation_image_omits_prompt_env_for_none(self):
        class FakeUUID:
            hex = "test-run-default-prompt"

        run_dir = Path(__file__).resolve().parents[1] / "runs" / FakeUUID.hex
        shutil.rmtree(run_dir, ignore_errors=True)

        try:
            with patch.object(main, "uuid4", return_value=FakeUUID()):
                with patch.object(main, "build_docker_image"):
                    with patch.object(main, "_wait_for_no_running_image_containers"):
                        with patch.object(main, "_validate_run_outputs"):
                            with patch.object(main.subprocess, "run") as run:
                                main.get_imitation_image(None, "https://example.com")

            docker_run_command = run.call_args.args[0]
            self.assertNotIn("UI_REPLICATE_PROMPT=", docker_run_command)
            self.assertIn("UI_REPLICATE_TARGET_IMAGE_URL=https://example.com", docker_run_command)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_get_imitation_image_removes_incomplete_run_directory_on_failure(self):
        class FakeUUID:
            hex = "test-run-incomplete"

        run_dir = Path(__file__).resolve().parents[1] / "runs" / FakeUUID.hex
        shutil.rmtree(run_dir, ignore_errors=True)

        with self.assertRaises(RuntimeError):
            with patch.object(main, "uuid4", return_value=FakeUUID()):
                with patch.object(main, "build_docker_image"):
                    with patch.object(main, "_wait_for_no_running_image_containers"):
                        with patch.object(main, "_run_docker_container"):
                            main.get_imitation_image(None, "https://example.com")

        self.assertFalse(run_dir.exists())

    def test_generate_imitation_runs_builds_once_and_runs_urls_in_order(self):
        target_image_urls = ["https://example.com/source"]
        expanded_urls = ["https://example.com/one.png", "https://example.com/two.png"]
        run_dirs = [Path(f"/tmp/run-{index}") for index in range(len(expanded_urls))]

        with patch.object(main, "build_docker_image") as build:
            with patch.object(main, "get_imitation_image", side_effect=run_dirs) as get_image:
                with patch.object(main, "expand_image_source_urls", return_value=expanded_urls) as expand_urls:
                    result = main.generate_imitation_runs(target_image_urls, prompt=None, score=False)

        self.assertEqual(result, run_dirs)
        build.assert_called_once_with()
        expand_urls.assert_called_once_with(target_image_urls)
        self.assertEqual(
            get_image.call_args_list,
            [
                unittest.mock.call(None, expanded_urls[0], build_image=False),
                unittest.mock.call(None, expanded_urls[1], build_image=False),
            ],
        )

    def test_generate_imitation_runs_rejects_empty_expanded_target_images(self):
        with patch.object(main, "build_docker_image"):
            with patch.object(main, "expand_image_source_urls", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "No target images were found"):
                    main.generate_imitation_runs(["https://huggingface.co/datasets/example/repo"], score=False)

    def test_resolve_single_image_url_returns_unchanged_url_for_non_dataset_sources(self):
        self.assertEqual(
            image_sources.resolve_single_image_url("https://example.com/image.png"),
            "https://example.com/image.png",
        )

    def test_load_rgb_image_array_converts_images_to_normalized_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "reference.png"
            Image.new("L", (1, 1), 128).save(image_path)

            result = scoring._load_rgb_image_array(image_path)

        self.assertEqual(result.shape, (1, 1, 3))
        np.testing.assert_allclose(result, np.full((1, 1, 3), 128 / 255, dtype=np.float32))

    def test_run_tests_returns_mse_and_reward_for_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            second = Path(tmp) / "second.jpg"
            Image.new("RGB", (1, 1), (0, 0, 0)).save(first)
            Image.new("RGB", (1, 1), (255, 255, 255)).save(second)

            result = scoring.run_tests_mse(first, second, sensitivity_score=0.5)

        self.assertEqual(result["mse"], 1.0)
        self.assertAlmostEqual(result["reward"], 1.0 / 3.0)

    def test_run_tests_lpips_reuses_loaded_image_arrays_for_learned_distance(self):
        observed_shapes = []

        def fake_lpips_distance(first, second):
            observed_shapes.extend([first.shape, second.shape])
            return 0.25

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            second = Path(tmp) / "second.jpg"
            Image.new("RGB", (2, 1), (0, 0, 0)).save(first)
            Image.new("RGB", (2, 1), (255, 255, 255)).save(second)

            result = scoring.run_tests_lpips(first, second, sensitivity_score=0.5, distance_fn=fake_lpips_distance)

        self.assertEqual(observed_shapes, [(1, 2, 3), (1, 2, 3)])
        self.assertEqual(result["lpips"], 0.25)
        self.assertAlmostEqual(result["reward"], 2.0 / 3.0)

    def test_mse_score_resizes_candidate_to_target_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.png"
            candidate = Path(tmp) / "candidate.png"
            Image.new("RGB", (4, 3), (10, 20, 30)).save(target)
            Image.new("RGB", (2, 2), (10, 20, 30)).save(candidate)

            result = scoring.mse_score(target, candidate)

        self.assertEqual(result, 0.0)

    def test_gpt4v_score_maps_pass_to_zero(self):
        with patch.object(scoring, "_llm_as_judge_verdict", return_value="PASS") as verdict:
            result = scoring.gpt4v_score("first.png", "second.png")

        self.assertEqual(result, 0.0)
        self.assertEqual(verdict.call_args.kwargs["model"], scoring.OPENROUTER_GPT4V_DEFAULT_MODEL)

    def test_gpt4v_score_maps_fail_to_one(self):
        with patch.object(scoring, "_llm_as_judge_verdict", return_value="FAIL"):
            result = scoring.gpt4v_score("first.png", "second.png")

        self.assertEqual(result, 1.0)

    def test_openrouter_chat_completion_loads_api_key_from_env_file(self):
        class FakeResponse(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

            with patch.dict(scoring.os.environ, {}, clear=True):
                with patch.object(scoring, "ENV_FILE", env_path):
                    with patch.object(scoring.urllib.request, "urlopen", return_value=FakeResponse(b'{"choices":[{"message":{"content":"PASS"}}]}')) as urlopen:
                        response = scoring._openrouter_chat_completion({"model": "openai/gpt-4.1-mini", "messages": []})

        self.assertEqual(response["choices"][0]["message"]["content"], "PASS")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")

    def test_response_message_text_handles_content_parts(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "PASS"},
                            {"type": "image_url", "image_url": {"url": "ignored"}},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(scoring._response_message_text(response_payload), "PASS")

    def test_extract_pass_fail_accepts_embedded_verdict(self):
        self.assertEqual(scoring._extract_pass_fail("Result: fail."), "FAIL")

    def test_validate_run_outputs_accepts_flat_screenshot_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "target.png").write_bytes(b"target")
            (run_dir / "ai-generated.png").write_bytes(b"generated")

            main._validate_run_outputs(run_dir)

    def test_main_accepts_score_equals_false(self):
        with patch.object(main, "generate_imitation_runs", return_value=[Path("/tmp/run")]) as generate:
            exit_code = main.main(["--score=False"])

        self.assertEqual(exit_code, 0)
        generate.assert_called_once_with(main.DEFAULT_TARGET_IMAGE_URLS, prompt=None, score=False)


if __name__ == "__main__":
    unittest.main()
