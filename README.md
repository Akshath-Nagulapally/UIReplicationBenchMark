# UI Replication Benchmark

This repository contains a cleaned-up benchmark harness for generating UI replication runs with OpenCode and visualizing the resulting screenshots.

Run generation from the repository root:

```bash
uv run main.py
```

Successful runs populate `runs/<run-id>/target.png` and `runs/<run-id>/ai-generated.png`.

Launch the visualization app from the repository root:

```bash
uv run visualize
```

The visualization backend lives in [`visualize`](/Users/aknag/Desktop/UIReplicationBenchMark/visualize), and the frontend lives in [`visualize/frontend`](/Users/aknag/Desktop/UIReplicationBenchMark/visualize/frontend). The app serves a JSON API from `runs/` and the frontend polls for updates automatically.
