import argparse
import html
import os
import re
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from visualize.scoring import llm_as_judge_score, lpips_score, mse_score

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BENCHMARK_DIR / "runs"
SCREENSHOTS_SUBDIR = "screenshots"
OUTPUT_FILE = BENCHMARK_DIR / "index.html"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
TARGET_IMAGE_NAME = "target.png"

# Add another metric by importing it above and appending it here.
SCORE_FUNCTIONS = (mse_score, lpips_score, llm_as_judge_score)


@dataclass
class ExampleResult:
    name: str
    target_path: Path
    ai_generated_path: Path
    scores: dict[str, float | str]
    average_rank: float | None = None


def metric_name(score_function):
    name = score_function.__name__
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
                scores[metric_name(score_function)] = score_function(target_path, ai_generated_path)
            except Exception as error:
                scores[metric_name(score_function)] = f"{type(error).__name__}: {error}"

        results.append(ExampleResult(name, target_path, ai_generated_path, scores))

    apply_average_ranks(results, [metric_name(function) for function in score_functions])
    return sorted(results, key=lambda result: (result.average_rank is None, result.average_rank or 0, result.name))


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


def render_html(results, score_functions, output_dir):
    score_names = [metric_name(function) for function in score_functions]
    headers = ["Avg Rank", "Run", "Target", "AI Generated", *[f"{name} Score" for name in score_names]]

    rows = "\n".join(render_row(index, result, score_names, output_dir) for index, result in enumerate(results, start=1))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Screenshot Similarity Experiment</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    body {{
      background: #f7f7f8;
      color: #19191c;
      margin: 0;
      padding: 32px;
    }}

    h1 {{
      margin: 0 0 8px;
    }}

    p {{
      color: #55555f;
      margin: 0 0 24px;
    }}

    table {{
      background: white;
      border-collapse: collapse;
      box-shadow: 0 8px 30px rgb(0 0 0 / 8%);
      width: 100%;
    }}

    th,
    td {{
      border-bottom: 1px solid #ececf0;
      padding: 12px;
      text-align: left;
      vertical-align: top;
    }}

    th {{
      background: #fbfbfc;
      color: #4a4a55;
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    img {{
      border: 1px solid #dedee5;
      border-radius: 8px;
      max-height: 280px;
      max-width: min(360px, 32vw);
    }}

    .number {{
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}

    .error {{
      color: #a40018;
      max-width: 260px;
    }}
  </style>
</head>
<body>
  <h1>Screenshot Similarity Experiment</h1>
  <p>Lower score is better. Runs are read from <code>runs/*</code>; legacy <code>runs/*/{SCREENSHOTS_SUBDIR}</code> directories are also supported.</p>
  <table>
    <thead>
      <tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr>
    </thead>
    <tbody>
      {rows or '<tr><td colspan="' + str(len(headers)) + '">No runs found.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""


def render_row(index, result, score_names, output_dir):
    score_cells = "".join(render_score_cell(result.scores[name]) for name in score_names)
    target_src = os.path.relpath(result.target_path, output_dir)
    ai_generated_src = os.path.relpath(result.ai_generated_path, output_dir)
    rank = format_score(result.average_rank) if result.average_rank is not None else "-"

    return f"""<tr>
  <td class="number">{rank}</td>
  <td>{html.escape(result.name)}</td>
  <td><img src="{html.escape(Path(target_src).as_posix())}" alt="Target for {html.escape(result.name)}"></td>
  <td><img src="{html.escape(Path(ai_generated_src).as_posix())}" alt="AI generated for {html.escape(result.name)}"></td>
  {score_cells}
</tr>"""


def render_score_cell(score):
    if isinstance(score, int | float):
        return f'<td class="number">{format_score(score)}</td>'

    return f'<td class="error">{html.escape(str(score))}</td>'


def format_score(score):
    return f"{score:.6g}"


def build_report(runs_dir=None, output_file=OUTPUT_FILE, score_functions=SCORE_FUNCTIONS):
    runs_dir = Path(runs_dir or RUNS_DIR)
    output_file = Path(output_file)

    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory does not exist: {runs_dir}")

    examples = discover_examples(runs_dir)
    results = score_examples(examples, score_functions)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_html(results, score_functions, output_file.parent), encoding="utf-8")
    return output_file


def serve_report(output_file, port):
    handler = partial(SimpleHTTPRequestHandler, directory=output_file.parent)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)

    print(f"Serving {output_file.name} at http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Render screenshot similarity experiment results.")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--no-serve", action="store_true", help="write the report without starting the local server")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = build_report(args.runs_dir, args.output)
    print(f"Wrote {output_file}", flush=True)

    if not args.no_serve:
        serve_report(output_file, args.port)


if __name__ == "__main__":
    main()
