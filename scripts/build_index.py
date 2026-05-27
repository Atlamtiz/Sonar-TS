"""Compute feature tables for every per-task database."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sonar_ts._paths import OFFLINE_CONFIG, TASKS_JSON, db_path_for
from sonar_ts.offline import compute_all_features
from sonar_ts.storage import TaskDatabase, FEATURE_TABLES


def _load_config() -> dict:
    with open(OFFLINE_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _features_present(db: TaskDatabase) -> bool:
    return all(db.table_row_count(t) > 0 for t in FEATURE_TABLES)


def _process_one(task_id: str, view_config: dict, alphabet: list[str],
                 *, rebuild: bool) -> tuple[str, str, dict]:
    if not db_path_for(task_id).exists():
        return task_id, "error", {"reason": "database missing — run load_benchmark first"}

    try:
        db = TaskDatabase(task_id)
        if _features_present(db) and not rebuild:
            counts = {t: db.table_row_count(t) for t in FEATURE_TABLES}
            db.close()
            return task_id, "skipped", counts

        raw_long = db.fetch_raw_long()
        feats = compute_all_features(raw_long, view_config, alphabet)
        for table_name, df in feats.items():
            if not df.empty:
                db.write_feature_table(table_name, df)
        counts = {t: db.table_row_count(t) for t in FEATURE_TABLES}
        db.close()
        return task_id, "built", counts
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[{task_id}] ERROR: {exc}\n{traceback.format_exc()}\n")
        return task_id, "error", {"reason": str(exc)}


def main() -> None:
    p = argparse.ArgumentParser(prog="build_index",
                                description="Compute SAX feature tables for every task database.")
    p.add_argument("--rebuild", action="store_true",
                   help="Recompute features even if all three tables are populated.")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers (default: from configs/offline.yaml).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N tasks (for testing).")
    args = p.parse_args()

    config = _load_config()
    alphabet = config["sax"]["alphabet"]
    view_config = config["views"]
    workers = args.workers if args.workers is not None else config.get("parallel", {}).get("workers", 8)

    with open(TASKS_JSON, encoding="utf-8") as f:
        tasks = json.load(f)
    task_ids = [t["id"] for t in tasks]
    if args.limit:
        task_ids = task_ids[: args.limit]

    print(f"Building features for {len(task_ids)} tasks")
    print(f"  alphabet  = {alphabet}")
    print(f"  views     = {list(view_config.keys())}")
    print(f"  workers   = {workers}  rebuild = {args.rebuild}")

    counts = {"built": 0, "skipped": 0, "error": 0}
    totals = {t: 0 for t in FEATURE_TABLES}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, tid, view_config, alphabet,
                               rebuild=args.rebuild): tid
                   for tid in task_ids}
        done = 0
        for fut in as_completed(futures):
            task_id, status, info = fut.result()
            counts[status] += 1
            if status in ("built", "skipped"):
                for t in FEATURE_TABLES:
                    totals[t] += info.get(t, 0)
            done += 1
            if done % 100 == 0 or done == len(task_ids):
                print(f"  [{done:>4d}/{len(task_ids)}]  "
                      f"built={counts['built']}  "
                      f"skipped={counts['skipped']}  "
                      f"error={counts['error']}")

    print("\nDone.")
    for t in FEATURE_TABLES:
        print(f"  {t:18s}  total rows: {totals[t]:>9,d}")
    if counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
