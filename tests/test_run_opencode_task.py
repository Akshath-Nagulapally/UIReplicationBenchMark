from __future__ import annotations

import importlib.util
import io
from queue import Queue
import subprocess
import tempfile
import unittest
from threading import Event
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "harnesses" / "opencode" / "run_opencode_task.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_opencode_task", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunOpencodeTaskTests(unittest.TestCase):
    def test_default_timeout_is_thirty_minutes(self):
        module = load_module()

        self.assertEqual(module.DEFAULT_TIMEOUT_SECONDS, 1800)

    def test_build_message_payload_includes_png_screenshot_part(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            screenshot_path = Path(tmp) / "screenshot.png"
            screenshot_path.write_bytes(b"png-bytes")

            payload = module.build_message_payload("build the app", screenshot_path=screenshot_path)

        self.assertEqual(payload["parts"][0], {"type": "text", "text": "build the app"})
        self.assertEqual(payload["parts"][1], {"type": "file", "mime": "image/png", "url": "data:image/png;base64,cG5nLWJ5dGVz"})

    def test_build_message_payload_requires_screenshot_path(self):
        module = load_module()

        with self.assertRaises(TypeError):
            module.build_message_payload("build the app")

    def test_cleanup_ports_deduplicates_requested_ports(self):
        module = load_module()
        calls: list[list[str]] = []
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def runner(command: list[str]):
            calls.append(command)
            return completed

        with patch.object(module.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}" if name == "lsof" else None):
            killed = module.kill_listeners_on_ports([5173, 5173, 8080], runner=runner)

        self.assertEqual(killed, [5173, 8080])
        self.assertEqual(calls, [["lsof", "-tiTCP:5173", "-sTCP:LISTEN"], ["lsof", "-tiTCP:8080", "-sTCP:LISTEN"]])

    def test_cleanup_uses_lsof_when_macos_fuser_does_not_support_kill(self):
        module = load_module()
        calls: list[list[str]] = []

        def runner(command: list[str]):
            calls.append(command)
            if command == ["fuser", "-k", "5173/tcp"]:
                return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="Unknown option: k")
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        with patch.object(module.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"fuser", "lsof"} else None):
            killed = module.kill_listeners_on_ports([5173], runner=runner)

        self.assertEqual(killed, [5173])
        self.assertEqual(calls, [["lsof", "-tiTCP:5173", "-sTCP:LISTEN"]])

    def test_main_requires_url(self):
        module = load_module()
        stderr = io.StringIO()

        with patch.object(module.sys, "stderr", stderr):
            with self.assertRaises(SystemExit) as error:
                module.main(["build the app"])

        self.assertEqual(error.exception.code, 2)
        self.assertIn("--image-url", stderr.getvalue())

    def test_download_reference_image_requests_the_provided_non_dataset_url(self):
        module = load_module()
        fake_response = io.BytesIO(b"png-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "target.png"
            with patch.object(module.urllib.request, "urlopen", return_value=fake_response) as urlopen:
                result = module.download_reference_image(
                    "https://example.com/reference.png",
                    output_path,
                )
            self.assertEqual(result, output_path)
            self.assertEqual(output_path.read_bytes(), b"png-bytes")
            request = urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.com/reference.png")

    def test_format_event_logs_tool_part_details(self):
        module = load_module()

        rendered = module.format_event_for_log(
            "message",
            {
                "type": "message.part.updated",
                "properties": {
                    "part": {
                        "type": "tool",
                        "tool": "bash",
                        "state": "error",
                        "command": "cat <<'EOF' > src/App.tsx\nexport default function App() {}\nEOF",
                        "stderr": "Permission denied",
                    }
                },
            },
        )

        self.assertIsNotNone(rendered)
        self.assertIn("tool=bash", rendered)
        self.assertIn("state=error", rendered)
        self.assertIn("command=", rendered)
        self.assertIn("Permission denied", rendered)

    def test_format_event_suppresses_image_data_in_parts(self):
        module = load_module()

        rendered = module.format_event_for_log(
            "message",
            {
                "type": "message.part.updated",
                "properties": {
                    "part": {
                        "type": "tool",
                        "tool": "write",
                        "input": {
                            "path": "src/App.tsx",
                            "url": "data:image/png;base64," + ("a" * 2000),
                        },
                    }
                },
            },
        )

        self.assertIsNotNone(rendered)
        self.assertIn("tool=write", rendered)
        self.assertIn("[image data suppressed]", rendered)
        self.assertNotIn("a" * 100, rendered)

    def test_session_matches_opencode_properties_wrapper(self):
        module = load_module()

        self.assertTrue(
            module.session_matches(
                {
                    "id": "evt_123",
                    "type": "session.idle",
                    "properties": {"sessionID": "ses_abc"},
                },
                "ses_abc",
            )
        )

    def test_wait_for_session_completion_uses_payload_event_type(self):
        module = load_module()
        event_queue = Queue()
        stop_stream = Event()
        final_message = {"info": {"id": "msg_final"}}
        event_queue.put(
            {
                "event": "message",
                "data": {
                    "id": "evt_idle",
                    "type": "session.idle",
                    "properties": {"sessionID": "ses_abc"},
                },
            }
        )

        with patch.object(module, "request_json", return_value=[final_message]) as request_json:
            response = module.wait_for_session_completion(
                "ses_abc",
                base_url="http://127.0.0.1:4096",
                directory=module.APP_DIR,
                queue=event_queue,
                stop_stream=stop_stream,
                timeout=1,
            )

        self.assertEqual(response, final_message)
        self.assertTrue(stop_stream.is_set())
        request_json.assert_called_once_with(
            "GET",
            "http://127.0.0.1:4096/session/ses_abc/message?limit=1",
            directory=module.APP_DIR,
            timeout=30,
        )

    def test_main_downloads_reference_and_captures_generated_screenshot(self):
        module = load_module()
        stdout = io.StringIO()
        reference_screenshot_path = module.DEFAULT_SCREENSHOT_PATH
        generated_screenshot_path = module.AI_GENERATED_SCREENSHOT_PATH
        dev_server = Mock()

        with patch.object(module, "download_reference_image", return_value=reference_screenshot_path) as download_reference_image:
            with patch.object(module, "capture_screenshot", return_value=generated_screenshot_path) as capture_screenshot:
                with patch.object(module, "run_task", return_value={"ok": True}) as run_task:
                    with patch.object(module, "start_dev_server", return_value=dev_server) as start_dev_server:
                        with patch.object(module, "wait_for_url") as wait_for_url:
                            with patch.object(module.sys, "stdout", stdout):
                                exit_code = module.main(["build the app", "--image-url", "https://example.com/reference.png"])

        self.assertEqual(exit_code, 0)
        download_reference_image.assert_called_once_with("https://example.com/reference.png", reference_screenshot_path)
        run_task.assert_called_once()
        start_dev_server.assert_called_once_with(module.APP_DIR)
        wait_for_url.assert_called_once_with("http://localhost:5173")
        capture_screenshot.assert_called_once_with("http://localhost:5173", module.AI_GENERATED_SCREENSHOT_PATH)
        dev_server.terminate.assert_called_once()
        dev_server.wait.assert_called_once_with(timeout=5)
        self.assertIn(str(generated_screenshot_path), stdout.getvalue())

    def test_main_downloads_reference_image_before_sending_task(self):
        module = load_module()
        stdout = io.StringIO()
        screenshot_path = module.DEFAULT_SCREENSHOT_PATH
        dev_server = Mock()

        with patch.object(module, "download_reference_image", return_value=screenshot_path) as download_reference_image:
            with patch.object(module, "capture_screenshot", return_value=module.AI_GENERATED_SCREENSHOT_PATH) as capture_screenshot:
                with patch.object(module, "run_task", return_value={"ok": True}) as run_task:
                    with patch.object(module, "start_dev_server", return_value=dev_server):
                        with patch.object(module, "wait_for_url"):
                            with patch.object(module.sys, "stdout", stdout):
                                exit_code = module.main(["build the app", "--image-url", "https://example.com/reference.png"])

        self.assertEqual(exit_code, 0)
        download_reference_image.assert_called_once_with("https://example.com/reference.png", screenshot_path)
        capture_screenshot.assert_called_once_with("http://localhost:5173", module.AI_GENERATED_SCREENSHOT_PATH)
        run_task.assert_called_once()
        self.assertIn('"ai_generated_screenshot"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
