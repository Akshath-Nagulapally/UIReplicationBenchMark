import fcntl
import argparse
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from uuid import uuid4


IMAGE_NAME = "ui-replication-benchmark"
CONTAINER_OUTPUT_DIR = "/workspace/output"
DEFAULT_TARGET_IMAGE_URL = "https://www.dropbox.com/scl/fi/4uh2rfbkgxchm8sezhya9/OutlookUI.png?rlkey=m6duzw3os8yiwpce9hfjx7q27&st=y46nyzt2&dl=1"
DEFAULT_TARGET_IMAGE_URLS = [
    DEFAULT_TARGET_IMAGE_URL,
    "https://www.dropbox.com/scl/fi/vai8f6m26bm6pfv5ezc3b/Anthropic.png?rlkey=ccun3ux59sf7roe0w8zcjh7j9&st=1sl32i5f&dl=1",
    "https://www.dropbox.com/scl/fi/14yzfzu9kufrr01149hoe/OpenAI.png?rlkey=mcwjsxjtom9h40klyr0xmrgj3&st=sg4c71ab&dl=1",
]
REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKER_LOCK_FILE = REPO_ROOT / ".docker-run.lock"
ENV_FILE = REPO_ROOT / ".env"
DOCKER_IMAGE_POLL_SECONDS = 5
DOCKER_RUN_TIMEOUT_SECONDS = None
REQUIRED_RUN_OUTPUTS = ("target.png", "ai-generated.png")
DOCKER_ENV_PASSTHROUGH_PREFIXES = ("OPENROUTER_",)
DOCKER_ENV_PASSTHROUGH_NAMES = {
    "OPENCODE_SERVER_USERNAME",
    "OPENCODE_SERVER_PASSWORD",
    "OPENCODE_TIMEOUT_SECONDS",
}


def _docker_run_timeout_seconds() -> int | None:
    raw = os.environ.get("UI_REPLICATE_DOCKER_TIMEOUT_SECONDS")
    if raw is None or raw == "":
        return DOCKER_RUN_TIMEOUT_SECONDS
    if raw.lower() in {"0", "false", "none", "no"}:
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError("UI_REPLICATE_DOCKER_TIMEOUT_SECONDS must be an integer number of seconds, or false") from error


def _load_env_file(env_file=None):
    env_file = ENV_FILE if env_file is None else env_file
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


@contextmanager
def _docker_run_lock():
    DOCKER_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DOCKER_LOCK_FILE.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _repo_paths():
    return REPO_ROOT, RUNS_DIR, DOCKERFILE


def build_docker_image():
    repo_root, _, dockerfile = _repo_paths()
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(dockerfile),
            "-t",
            IMAGE_NAME,
            str(repo_root),
        ],
        check=True,
    )


def _running_container_ids_for_image():
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-q",
            "--filter",
            f"ancestor={IMAGE_NAME}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [container_id for container_id in result.stdout.splitlines() if container_id]


def _wait_for_no_running_image_containers():
    while _running_container_ids_for_image():
        time.sleep(DOCKER_IMAGE_POLL_SECONDS)


def _validate_run_outputs(run_dir):
    missing = [
        path.relative_to(run_dir)
        for path in (run_dir / output_name for output_name in REQUIRED_RUN_OUTPUTS)
        if not path.is_file()
    ]
    if missing:
        missing_files = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"run did not produce required screenshot files: {missing_files}")


def _stop_container(container_name):
    subprocess.run(["docker", "stop", container_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_docker_container(run_command, container_name):
    try:
        subprocess.run(run_command, check=True, timeout=_docker_run_timeout_seconds())
    except subprocess.TimeoutExpired:
        _stop_container(container_name)
        raise


def _normalize_dropbox_direct_link(url):
    if "dropbox.com" not in url:
        return url

    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.pop("raw", None)
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def get_imitation_image(prompt, target_image_url, *, build_image=True):
    _load_env_file()
    normalized_target_image_url = _normalize_dropbox_direct_link(target_image_url or DEFAULT_TARGET_IMAGE_URL)
    _, runs_dir, _ = _repo_paths()
    run_dir = runs_dir / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    container_name = f"{IMAGE_NAME}-{run_dir.name}"

    try:
        if build_image:
            build_docker_image()

        run_command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "-v",
            f"{run_dir}:{CONTAINER_OUTPUT_DIR}",
            "-e",
            f"UI_REPLICATE_TARGET_IMAGE_URL={normalized_target_image_url}",
        ]
        run_command.extend(_docker_env_args())

        if prompt is not None:
            run_command.extend(["-e", f"UI_REPLICATE_PROMPT={prompt}"])

        opencode_timeout = os.environ.get("OPENCODE_TIMEOUT_SECONDS")
        if opencode_timeout:
            run_command.extend(["-e", f"OPENCODE_TIMEOUT_SECONDS={opencode_timeout}"])

        if sys.stdin.isatty():
            run_command.append("-it")

        run_command.append(IMAGE_NAME)
        with _docker_run_lock():
            _wait_for_no_running_image_containers()
            _run_docker_container(run_command, container_name)
        _validate_run_outputs(run_dir)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir


def _docker_env_args():
    args = []
    for key, value in sorted(os.environ.items()):
        should_pass = key in DOCKER_ENV_PASSTHROUGH_NAMES or any(
            key.startswith(prefix) for prefix in DOCKER_ENV_PASSTHROUGH_PREFIXES
        )
        if should_pass:
            args.extend(["-e", f"{key}={value}"])
    return args


def generate_imitation_runs(target_image_urls=DEFAULT_TARGET_IMAGE_URLS, prompt=None, *, score=True):
    build_docker_image()
    run_dirs = []
    for target_image_url in target_image_urls:
        run_dirs.append(get_imitation_image(prompt, target_image_url, build_image=False))
    if score:
        build_score_report()
    return run_dirs


def build_score_report(runs_dir=RUNS_DIR):
    from visualize.visualize_results import build_report

    return build_report(runs_dir=runs_dir)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Generate UI replication runs with the OpenCode harness.")
    parser.add_argument(
        "--target-image-url",
        action="append",
        default=None,
        help="Target image URL to replicate. May be provided multiple times. Defaults to the built-in target set.",
    )
    parser.add_argument("--prompt", default=None, help="Prompt sent to OpenCode. Defaults to the harness prompt.")
    parser.add_argument(
        "--score",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Build the visualization report after generation. Defaults to true.",
    )
    parser.add_argument("--no-score", action="store_false", dest="score", help="Skip report generation.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    target_image_urls = args.target_image_url or DEFAULT_TARGET_IMAGE_URLS
    try:
        run_dirs = generate_imitation_runs(target_image_urls, prompt=args.prompt, score=args.score)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for run_dir in run_dirs:
        print(run_dir)
    if args.score:
        print(REPO_ROOT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
