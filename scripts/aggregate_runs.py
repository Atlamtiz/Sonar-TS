#!/usr/bin/env python3
"""Average one or more evaluation runs into the row of the main results table.

Each run directory holds the `records.jsonl` written by `cold_start_stream.eval`
(one line per task with `task_id`, `category`, `level`, `score`). Scores are
averaged per task over the runs, then per task category, per level, and over all
tasks (the mean over tasks, which weights every category by its size).

    python scripts/aggregate_runs.py runs/main/r1 runs/main/r2 runs/main/r3
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ["Atomic Retrieval", "Sliding Window", "Shape Identification", "Periodicity Detection",
         "Subsequence Matching", "Composite Trend", "Contextual Anomaly", "Causal Anomaly", "Insight Synthesis"]
ABBR = {"Atomic Retrieval": "AR", "Sliding Window": "SW", "Shape Identification": "SI", "Periodicity Detection": "PD",
        "Subsequence Matching": "SM", "Composite Trend": "CT", "Contextual Anomaly": "CxA", "Causal Anomaly": "CsA",
        "Insight Synthesis": "IS"}


def main(run_dirs: list[str]) -> None:
    per_task: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, tuple[str, int]] = {}
    for d in run_dirs:
        for line in open(Path(d) / "records.jsonl", encoding="utf-8"):
            r = json.loads(line)
            per_task[r["task_id"]].append(float(r["score"]))
            meta[r["task_id"]] = (r["category"], int(r["level"]))
    task_mean = {t: sum(v) / len(v) for t, v in per_task.items()}
    by_cat, by_level = defaultdict(list), defaultdict(list)
    for t, m in task_mean.items():
        cat, lvl = meta[t]
        by_cat[cat].append(m)
        by_level[lvl].append(m)
    print(f"runs: {len(run_dirs)}   tasks: {len(task_mean)}")
    print("task      n     score")
    for cat in ORDER:
        if cat in by_cat:
            print(f"{ABBR[cat]:5s} {len(by_cat[cat]):5d}   {sum(by_cat[cat]) / len(by_cat[cat]):.4f}")
    for lvl in sorted(by_level):
        print(f"L{lvl}    {len(by_level[lvl]):5d}   {sum(by_level[lvl]) / len(by_level[lvl]):.4f}")
    print(f"Avg   {len(task_mean):5d}   {sum(task_mean.values()) / len(task_mean):.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
