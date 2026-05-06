from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_VIEWPORT_WIDTH = 1440
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_OUTPUT_PATH = Path("screenshots") / "screenshot.png"


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        return url
    return f"http://{url}"


def capture_screenshot(
    url: str,
    output_path: Path,
    *,
    width: int = DEFAULT_VIEWPORT_WIDTH,
    height: int = DEFAULT_VIEWPORT_HEIGHT,
    playwright=None,
) -> Path:
    close_playwright = None
    if playwright is None:
        from playwright.sync_api import sync_playwright

        close_playwright = sync_playwright().start()
        playwright = close_playwright

    output_path.parent.mkdir(parents=True, exist_ok=True)
    browser = playwright.chromium.launch()
    try:
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(normalize_url(url), wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        page.screenshot(path=str(output_path), full_page=False)
    finally:
        browser.close()
        if close_playwright is not None:
            close_playwright.stop()

    return output_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a deterministic full-page screenshot for a URL.")
    parser.add_argument("url", help="URL to screenshot, for example localhost:5173 or https://example.com")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Screenshot output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWPORT_WIDTH, help="Viewport width before capture.")
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWPORT_HEIGHT, help="Viewport height before capture.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output_path = capture_screenshot(args.url, args.output, width=args.width, height=args.height)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
