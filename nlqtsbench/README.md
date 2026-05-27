# NLQTSBench

The natural-language → time-series query benchmark used in **"Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases"** . 1,153 tasks across nine sub-tasks spanning four levels of querying difficulty.

## What's in this directory

| File / path           | What it is                                                  |
|-----------------------|-------------------------------------------------------------|
| `tasks.json`          | 1,153 task records (id, level, category, sub-task, question, ground-truth, eval_metric, channel, ts_data_path). |
| `predict_perfect.json`| Submission file containing the *ground truth* as predictions — useful for sanity-checking the evaluator (`python main.py` should score 1.0). |
| `ts_data/`            | One CSV per task (~1.7 GB total). **Not in this repository** — see "Downloading the CSVs" below. |

## Downloading the CSVs

The raw time-series CSVs (≈ 1.7 GB) are hosted on HuggingFace at [`mrtan/NLQTSBench`](https://huggingface.co/datasets/mrtan/NLQTSBench). A one-liner from the repository root pulls them into the expected location:

```bash
python scripts/download_dataset.py
```

The script:

1. Downloads `ts_data/**` from the HuggingFace dataset.
2. Places the files at `nlqtsbench/ts_data/<task_id>.csv`.

The framework's loader (`scripts/load_benchmark.py`) reads each CSV and materialises it into a per-task SQLite database under `databases/`.

## Evaluation

Each task is scored by **one of five metrics**, chosen automatically per sub-task. All scoring logic lives in [`sonar_ts/evaluator.py`](../sonar_ts/evaluator.py).

### Metrics

| Metric    | Sub-tasks                                                                                                          | What it measures                                                              |
|-----------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `rel_acc` | Global Aggregation, Periodicity Detection                                                                          | Relative accuracy on a scalar: `max(0, 1 − |p − g| / (|g| + ε))`              |
| `hit`     | Temporal Localization                                                                                              | Binary exact match on an ISO timestamp                                        |
| `iou`     | Interval Discovery, Sliding Window, Shape Identification, Subsequence Matching, Contextual Anomaly, Causal Anomaly | Intersection-over-Union of two `[start, end]` time intervals (seconds)        |
| `set_f1`  | Composite Trend                                                                                                    | F1 on the unordered set of returned date strings                              |
| `report`  | Insight Synthesis                                                                                                  | Composite: `0.4·trend + 0.3·interval + 0.2·adjective + 0.1·outlier`           |


### Run it standalone (recommended for a foreign submission)

If you have a `predict.json` produced by your own model (or any external system) and just want to score it against the benchmark, run the evaluator as a standalone module. **No LLM pipeline is invoked.**

```bash
# Score and print results
python -m sonar_ts.evaluator \
    --tasks   nlqtsbench/tasks.json \
    --predict path/to/your_predict.json
```
### Run it as part of the full pipeline

The repository-root `main.py` runs the Sonar-TS LLM pipeline AND scores the result in one shot. Use this when reproducing the paper's numbers from scratch:

```bash
python main.py
```

It writes:

* `output/predict.json` — the framework's predictions
* `output/summary.json` — the same paper-aligned table as the standalone evaluator
* `output/per_task.json` — one row per task with `prediction`, `score`, and `breakdown` (sub-scores for the `report` metric)

### Use it programmatically

```python
from sonar_ts.evaluator import score_one, aggregate

# Score a single prediction
score, breakdown = score_one("iou", pred="...", gt="...")

# Aggregate per-row results into paper-aligned buckets
agg = aggregate(per_row_results)
print(agg["by_category"]["Composite Trend"]["avg"])
```

## Submission format

A submission is a JSON array of `{id, prediction}` records, one per row in `tasks.json` (matched by 0-indexed `id`):

```json
[
  {"id": 0,    "prediction": "0.479"},
  {"id": 1,    "prediction": "['2022-09-11 19:15:00', '2022-09-20 13:00:00']"},
  {"id": 2,    "prediction": "2023-04-15 03:30:00"},
  ...
  {"id": 1152, "prediction": "..."}
]
```

All predictions are strings; the evaluator parses each one into the type its metric expects. Missing IDs are scored 0.

