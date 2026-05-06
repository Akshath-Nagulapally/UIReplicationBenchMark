from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "image_collection_utilities" / "screenshot.py"


def load_module():
    assert SCRIPT_PATH.exists(), "screenshot.py should exist"
    spec = importlib.util.spec_from_file_location("screenshot", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScreenshotTests(unittest.TestCase):
    def test_normalize_url_adds_http_scheme_for_localhost(self):
        module = load_module()

        self.assertEqual(module.normalize_url("localhost:5173"), "http://localhost:5173")

    def test_capture_screenshot_uses_fixed_viewport_only(self):
        module = load_module()
        screenshot = Mock()
        page = Mock(screenshot=screenshot)
        context = Mock()
        context.new_page.return_value = page
        browser = Mock()
        browser.new_context.return_value = context
        chromium = Mock()
        chromium.launch.return_value = browser
        playwright = Mock(chromium=chromium)
        output = Path("page.png")

        result = module.capture_screenshot(
            "localhost:5173",
            output,
            playwright=playwright,
            width=1440,
            height=900,
        )

        self.assertEqual(result, output)
        browser.new_context.assert_called_once_with(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto.assert_called_once_with("http://localhost:5173", wait_until="networkidle", timeout=30000)
        screenshot.assert_called_once_with(path=str(output), full_page=False)
        browser.close.assert_called_once()

    def test_main_writes_to_default_screenshot_path(self):
        module = load_module()
        stderr = io.StringIO()

        expected_path = Path("screenshots") / "screenshot.png"

        with patch.object(module, "capture_screenshot", return_value=expected_path) as capture_screenshot:
            with patch.object(module.sys, "stderr", stderr):
                exit_code = module.main(["http://example.com"])

        self.assertEqual(exit_code, 0)
        capture_screenshot.assert_called_once_with(
            "http://example.com",
            expected_path,
            width=module.DEFAULT_VIEWPORT_WIDTH,
            height=module.DEFAULT_VIEWPORT_HEIGHT,
        )
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
