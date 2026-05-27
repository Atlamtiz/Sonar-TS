"""Load NLQTSBench CSVs into per-task SQLite databases."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sonar_ts._paths import DATABASES_DIR, TASKS_JSON, db_path_for, csv_path_for
from sonar_ts.storage import TaskDatabase


def _process_one(task_id: str, *, rebuild: bool) -> tuple[str, str, int]:
    db_path = db_path_for(task_id)
    if db_path.exists() and not rebuild:
        try:
            db = TaskDatabase(task_id)
            n = db.table_row_count("raw_data")
            db.close()
            if n > 0:
                return task_id, "skipped", n
        except Exception:  # noqa: BLE001
            pass

    try:
        db = TaskDatabase.from_csv(task_id, overwrite=True)
        n = db.table_row_count("raw_data")
        db.close()
        return task_id, "created", n
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[{task_id}] ERROR: {exc}\n{traceback.format_exc()}\n")
        return task_id, "error", 0


def main() -> None:
    p = argparse.ArgumentParser(prog="load_benchmark",
                                description="Load NLQTSBench CSVs into per-task SQLite files.")
    p.add_argument("--rebuild", action="store_true",
                   help="Overwrite existing databases instead of skipping them.")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel workers (default: 8).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N tasks (for testing).")
    args = p.parse_args()

    with open(TASKS_JSON, encoding="utf-8") as f:
        tasks = json.load(f)
    task_ids = [t["id"] for t in tasks]
    if args.limit:
        task_ids = task_ids[: args.limit]

    DATABASES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(task_ids)} tasks  →  {DATABASES_DIR}/")
    print(f"  rebuild={args.rebuild}  workers={args.workers}")

    counts = {"created": 0, "skipped": 0, "error": 0}
    total_rows = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, tid, rebuild=args.rebuild): tid
                   for tid in task_ids}
        done = 0
        for fut in as_completed(futures):
            task_id, status, n = fut.result()
            counts[status] += 1
            total_rows += n
            done += 1
            if done % 100 == 0 or done == len(task_ids):
                print(f"  [{done:>4d}/{len(task_ids)}]  "
                      f"created={counts['created']}  "
                      f"skipped={counts['skipped']}  "
                      f"error={counts['error']}")

    print(f"\nDone. raw_data rows across all databases: {total_rows:,}")
    if counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
