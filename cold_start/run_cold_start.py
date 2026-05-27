"""Cold-start CLI: drive the per-subtask iteration loop and persist results."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from sonar_ts import _paths as paths_mod
from sonar_ts.llm import load_api_keys

from .orchestrator import cold_start_one_subtask


ALL_SUBTASKS = [
    "Global Aggregation",
    "Temporal Localization",
    "Interval Discovery",
    "Sliding Window",
    "Shape Identification",
    "Periodicity Detection",
    "Subsequence Matching",
    "Composite Trend",
    "Contextual Anomaly",
    "Causal Anomaly",
    "Insight Synthesis",
]


TRAIN_DIR        = _HERE / "train_data"
TRAIN_TASKS_JSON = TRAIN_DIR / "tasks.json"
TRAIN_DATABASES  = TRAIN_DIR / "databases"
TRAIN_TS_DATA    = TRAIN_DIR / "ts_data"

DISCOVERED_DIR = _HERE / "discovered_skills"
TRACES_DIR     = _HERE / "traces"
SUMMARY_JSON   = _HERE / "summary.json"


def _safe_filename(subtask: str) -> str:
    return subtask.replace(" ", "_")


def main() -> None:
    p = argparse.ArgumentParser(prog="run_cold_start",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subtask", default=None,
                   help="Only run this sub-task (default: all 11).")
    p.add_argument("--max-rounds", type=int, default=3,
                   help="Max reflect/draft/critic/revise rounds (default: 3).")
    p.add_argument("--sample", type=int, default=0,
                   help="Cap training tasks per sub-task to N (default: 0 = "
                        "use all).")
    p.add_argument("--workers", type=int, default=10,
                   help="LLM worker threads for verify pass (default: 10).")
    p.add_argument("--rebuild", action="store_true",
                   help="Wipe traces/ and discovered_skills/ first.")
    args = p.parse_args()

    if not TRAIN_TASKS_JSON.is_file():
        raise SystemExit(
            f"training tasks not found at {TRAIN_TASKS_JSON}.\n"
            f"Run first:  python -m cold_start.download_train_data"
        )

    paths_mod.TASKS_JSON    = TRAIN_TASKS_JSON
    paths_mod.DATABASES_DIR = TRAIN_DATABASES
    paths_mod.TS_DATA_DIR   = TRAIN_TS_DATA

    with open(_FRAMEWORK_ROOT / "configs" / "online.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_cfg = config["llm"]
    exec_cfg = {
        "max_regenerations": config["executor"]["max_regenerations"],
        "timeout_seconds":   config["executor"]["timeout_seconds"],
        "shape_rules":       config["shape_rules"],
    }
    api_keys = load_api_keys(llm_cfg["api_keys_file"])

    if args.rebuild:
        for d in (TRACES_DIR, DISCOVERED_DIR):
            if d.exists():
                shutil.rmtree(d)
    DISCOVERED_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = json.loads(TRAIN_TASKS_JSON.read_text(encoding="utf-8"))
    by_subtask: dict[str, list] = {st: [] for st in ALL_SUBTASKS}
    for t in all_tasks:
        st = t.get("subtask")
        if st in by_subtask:
            by_subtask[st].append(t)

    targets = [args.subtask] if args.subtask else ALL_SUBTASKS
    if args.subtask and args.subtask not in ALL_SUBTASKS:
        raise SystemExit(f"unknown subtask {args.subtask!r}. "
                         f"choices: {ALL_SUBTASKS}")

    print("=" * 78)
    print("  Sonar-TS automated cold-start")
    print("=" * 78)
    print(f"  Subtasks   : {len(targets)}")
    print(f"  Train root : {TRAIN_DIR}")
    print(f"  Max rounds : {args.max_rounds}")
    print(f"  Sample cap : {'(no cap)' if not args.sample else args.sample}")
    print(f"  Workers    : {args.workers}")
    print(f"  Out skills : {DISCOVERED_DIR}")
    print(f"  Out traces : {TRACES_DIR}")
    print()

    summary: dict[str, list] = {}
    for subtask in targets:
        tasks = by_subtask.get(subtask, [])
        if args.sample:
            tasks = tasks[: args.sample]
        if not tasks:
            print(f"[skip] {subtask}: no training tasks available")
            continue
        eval_metric = tasks[0]["eval_metric"]
        print(f"--- {subtask} ({eval_metric}, N={len(tasks)}) ---")
        rounds = cold_start_one_subtask(
            subtask=subtask, eval_metric=eval_metric,
            train_tasks=tasks,
            llm_cfg=llm_cfg, exec_cfg=exec_cfg, api_keys=api_keys,
            max_rounds=args.max_rounds,
            workers=args.workers,
            traces_dir=TRACES_DIR / _safe_filename(subtask),
            out_skill_path=DISCOVERED_DIR / f"{_safe_filename(subtask)}.md",
        )
        summary[subtask] = [{"v": r.v, "score": r.score,
                              "n_examples": r.n_examples} for r in rounds]
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2),
                                  encoding="utf-8")

    print("\n" + "=" * 78)
    print("  Cold-start finished")
    print("=" * 78)
    for st, rounds in summary.items():
        if not rounds:
            print(f"  {st:24s}  (no rounds)")
            continue
        best = max(rounds, key=lambda r: r["score"])
        print(f"  {st:24s}  best v{best['v']}: score = {best['score']:.4f}")
    print(f"\nSummary  : {SUMMARY_JSON}")
    print(f"Skills   : {DISCOVERED_DIR}")
    print(f"Traces   : {TRACES_DIR}")


if __name__ == "__main__":
    main()
