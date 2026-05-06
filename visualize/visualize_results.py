import argparse
import json
import re
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from visualize.scoring import gpt4v_score

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BENCHMARK_DIR / "runs"
SCREENSHOTS_SUBDIR = "screenshots"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
TARGET_IMAGE_NAME = "target.png"

SCORE_FUNCTIONS = (gpt4v_score,)
_SCORE_CACHE = {}


@dataclass
class ExampleResult:
    name: str
    target_path: Path
    ai_generated_path: Path
    scores: dict[str, float | str]
    average_rank: float | None = None


def metric_name(score_function):
    name = score_function.__name__
    if name == "gpt4v_score":
        return "GPTVISION"
    if name.endswith("_score"):
        name = name[: -len("_score")]
    return name.upper()


def discover_examples(runs_dir):
    examples = []

    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), key=natural_sort_key):
        example = discover_example(run_dir)
        if example is not None:
            examples.append(example)

    return examples


def discover_example(run_dir):
    for image_dir in (run_dir, run_dir / SCREENSHOTS_SUBDIR):
        if not image_dir.is_dir():
            continue

        image_paths = discover_image_files(image_dir)
        target_path = image_dir / TARGET_IMAGE_NAME
        if target_path not in image_paths:
            continue

        generated_paths = [path for path in image_paths if path != target_path]
        if len(generated_paths) == 1:
            return (run_dir.name, target_path, generated_paths[0])

    return None


def discover_image_files(directory):
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_sort_key,
    )


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def score_examples(examples, score_functions):
    results = []

    for name, target_path, ai_generated_path in examples:
        scores = {}

        for score_function in score_functions:
            try:
                scores[metric_name(score_function)] = cached_score(score_function, target_path, ai_generated_path)
            except Exception as error:
                scores[metric_name(score_function)] = f"{type(error).__name__}: {error}"

        results.append(ExampleResult(name, target_path, ai_generated_path, scores))

    apply_average_ranks(results, [metric_name(function) for function in score_functions])
    return sorted(results, key=lambda result: (result.average_rank is None, result.average_rank or 0, result.name))


def cached_score(score_function, target_path, ai_generated_path):
    target_stat = target_path.stat()
    ai_generated_stat = ai_generated_path.stat()
    cache_key = (
        metric_name(score_function),
        target_path,
        target_stat.st_mtime_ns,
        target_stat.st_size,
        ai_generated_path,
        ai_generated_stat.st_mtime_ns,
        ai_generated_stat.st_size,
    )
    if cache_key not in _SCORE_CACHE:
        _SCORE_CACHE[cache_key] = score_function(target_path, ai_generated_path)
    return _SCORE_CACHE[cache_key]


def apply_average_ranks(results, metric_names):
    rank_totals = {result.name: [] for result in results}

    for name in metric_names:
        score_rows = [
            (result, result.scores[name])
            for result in results
            if isinstance(result.scores.get(name), int | float)
        ]

        for rank, (result, _score) in enumerate(sorted(score_rows, key=lambda row: row[1]), start=1):
            rank_totals[result.name].append(rank)

    for result in results:
        ranks = rank_totals[result.name]
        if ranks:
            result.average_rank = sum(ranks) / len(ranks)


def format_score(score):
    return f"{score:.6g}"


def build_results_payload(runs_dir=None, score_functions=SCORE_FUNCTIONS):
    runs_dir = Path(runs_dir or RUNS_DIR)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory does not exist: {runs_dir}")

    examples = discover_examples(runs_dir)
    results = score_examples(examples, score_functions)
    score_names = [metric_name(function) for function in score_functions]
    return {
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "runsDir": str(runs_dir.relative_to(BENCHMARK_DIR)) if runs_dir.is_relative_to(BENCHMARK_DIR) else str(runs_dir),
        "scoreNames": score_names,
        "results": [serialize_result(result) for result in results],
    }


def serialize_result(result):
    return {
        "name": result.name,
        "averageRank": result.average_rank,
        "targetImageUrl": image_url_for_path(result.target_path),
        "candidateImageUrl": image_url_for_path(result.ai_generated_path),
        "scores": result.scores,
    }


def image_url_for_path(path):
    relative_path = path.relative_to(BENCHMARK_DIR)
    return f"/{relative_path.as_posix()}"


def build_report(runs_dir=None, output_file=None, score_functions=SCORE_FUNCTIONS):
    build_results_payload(runs_dir=runs_dir, score_functions=score_functions)
    if not FRONTEND_INDEX.is_file():
        raise FileNotFoundError(f"frontend entrypoint does not exist: {FRONTEND_INDEX}")
    return FRONTEND_INDEX


class VisualizeRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, frontend_dir, repo_root, runs_dir, score_functions, **kwargs):
        self.frontend_dir = Path(frontend_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.runs_dir = Path(runs_dir).resolve()
        self.score_functions = score_functions
        super().__init__(*args, directory=str(self.frontend_dir), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/results":
            payload = build_results_payload(runs_dir=self.runs_dir, score_functions=self.score_functions)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def translate_path(self, path):
        requested_path = urlparse(path).path
        if requested_path.startswith("/runs/"):
            return str(self._safe_path(self.repo_root, requested_path.removeprefix("/")))

        relative_path = requested_path.lstrip("/") or "index.html"
        candidate = self._safe_path(self.frontend_dir, relative_path)
        if candidate.is_file():
            return str(candidate)
        return str(self.frontend_dir / "index.html")

    def _safe_path(self, root, relative_path):
        decoded_path = unquote(relative_path)
        candidate = (root / decoded_path).resolve()
        if candidate == root or root in candidate.parents:
            return candidate
        return root / "index.html"


def serve_report(runs_dir, port, score_functions=SCORE_FUNCTIONS):
    handler = partial(
        VisualizeRequestHandler,
        frontend_dir=FRONTEND_DIR,
        repo_root=BENCHMARK_DIR,
        runs_dir=Path(runs_dir or RUNS_DIR),
        score_functions=score_functions,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)

    print(f"Serving visualize app at http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Render screenshot similarity experiment results.")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="deprecated; frontend is now served from visualize/frontend")
    parser.add_argument("--no-serve", action="store_true", help="precompute scores without starting the local server")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    frontend_entrypoint = build_report(args.runs_dir, args.output)
    print(f"Prepared {frontend_entrypoint}", flush=True)

    if not args.no_serve:
        serve_report(args.runs_dir, args.port)


if __name__ == "__main__":
    main()
