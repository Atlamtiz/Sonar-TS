"""Download the cold-start training set and build per-task SQLite + SAX tables."""

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


TRAIN_DIR        = _HERE / "train_data"
TRAIN_TASKS_JSON = TRAIN_DIR / "tasks.json"
TRAIN_TS_DATA    = TRAIN_DIR / "ts_data"
TRAIN_DATABASES  = TRAIN_DIR / "databases"

HF_REPO_ID   = "mrtan/NLQTSBench-train"
HF_REPO_TYPE = "dataset"
HF_PATTERNS  = ["tasks.json", "ts_data/**"]


def _stage1_download(rebuild: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "    pip install huggingface_hub"
        )

    if rebuild:
        if TRAIN_TASKS_JSON.exists():
            TRAIN_TASKS_JSON.unlink()
        if TRAIN_TS_DATA.exists():
            shutil.rmtree(TRAIN_TS_DATA)

    if TRAIN_TASKS_JSON.exists() and TRAIN_TS_DATA.exists() \
            and any(TRAIN_TS_DATA.glob("*.csv")):
        n = sum(1 for _ in TRAIN_TS_DATA.glob("*.csv"))
        print(f"[stage 1] already present: tasks.json + {n} CSVs "
              f"(use --rebuild to re-download)")
        return

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[stage 1] downloading {HF_REPO_ID} → {TRAIN_DIR} …")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        allow_patterns=HF_PATTERNS,
        local_dir=str(TRAIN_DIR),
    )
    n_tasks = len(json.loads(TRAIN_TASKS_JSON.read_text(encoding="utf-8")))
    n_csv   = sum(1 for _ in TRAIN_TS_DATA.glob("*.csv"))
    print(f"[stage 1] tasks.json : {n_tasks} tasks")
    print(f"[stage 1] ts_data/   : {n_csv} CSVs")


def _stage2_build_databases(rebuild: bool) -> None:
    from sonar_ts import _paths as paths_mod
    from sonar_ts.storage import TaskDatabase

    if rebuild and TRAIN_DATABASES.exists():
        shutil.rmtree(TRAIN_DATABASES)
    TRAIN_DATABASES.mkdir(parents=True, exist_ok=True)

    paths_mod.TASKS_JSON    = TRAIN_TASKS_JSON
    paths_mod.DATABASES_DIR = TRAIN_DATABASES
    paths_mod.TS_DATA_DIR   = TRAIN_TS_DATA

    tasks = json.loads(TRAIN_TASKS_JSON.read_text(encoding="utf-8"))
    print(f"[stage 2] loading {len(tasks)} tasks → {TRAIN_DATABASES} …")

    n_ok = n_skip = n_err = 0
    for t in tasks:
        task_id = t["id"]
        db_path = TRAIN_DATABASES / f"{task_id}.sqlite"
        if db_path.exists() and not rebuild:
            try:
                db = TaskDatabase(task_id, db_path=db_path)
                if db.table_row_count("raw_data") > 0:
                    db.close()
                    n_skip += 1
                    continue
                db.close()
            except Exception:  # noqa: BLE001
                pass
        csv_path = TRAIN_TS_DATA / f"{task_id}.csv"
        if not csv_path.is_file():
            sys.stderr.write(f"  skip {task_id}: csv missing\n")
            n_err += 1
            continue
        try:
            db = TaskDatabase.from_csv(task_id, csv_path,
                                        db_path=db_path, overwrite=True)
            db.close()
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"  error {task_id}: "
                             f"{type(exc).__name__}: {exc}\n")
            n_err += 1
    print(f"[stage 2] loaded {n_ok}, skipped {n_skip}, errors {n_err}")


def _stage3_build_features(rebuild: bool) -> None:
    from sonar_ts import _paths as paths_mod
    from sonar_ts.storage import TaskDatabase, FEATURE_TABLES
    from sonar_ts.offline import compute_all_features

    paths_mod.TASKS_JSON    = TRAIN_TASKS_JSON
    paths_mod.DATABASES_DIR = TRAIN_DATABASES
    paths_mod.TS_DATA_DIR   = TRAIN_TS_DATA

    cfg = yaml.safe_load((_FRAMEWORK_ROOT / "configs" / "offline.yaml")
                          .read_text(encoding="utf-8"))
    views    = cfg["views"]
    alphabet = cfg["sax"]["alphabet"]

    tasks = json.loads(TRAIN_TASKS_JSON.read_text(encoding="utf-8"))
    print(f"[stage 3] building features for {len(tasks)} databases …")

    n_ok = n_skip = n_err = 0
    for t in tasks:
        task_id = t["id"]
        db_path = TRAIN_DATABASES / f"{task_id}.sqlite"
        if not db_path.exists():
            sys.stderr.write(f"  skip {task_id}: db missing\n")
            n_err += 1
            continue
        try:
            db = TaskDatabase(task_id, db_path=db_path)
            already_built = all(db.table_row_count(tbl) > 0
                                for tbl in FEATURE_TABLES)
            if already_built and not rebuild:
                db.close()
                n_skip += 1
                continue
            raw_long = db.fetch_raw_long()
            feats = compute_all_features(raw_long, views, alphabet)
            for table_name, df in feats.items():
                if not df.empty:
                    db.write_feature_table(table_name, df)
            db.close()
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"  error {task_id}: "
                             f"{type(exc).__name__}: {exc}\n")
            n_err += 1
    print(f"[stage 3] built {n_ok}, skipped {n_skip}, errors {n_err}")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="download_train_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true",
                   help="Wipe local artefacts and re-run every stage.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the HuggingFace download (rebuild dbs/features only).")
    args = p.parse_args()

    print("Cold-start training set — downloader")
    print("=" * 60)
    print(f"  HF repo    : {HF_REPO_ID}")
    print(f"  Local dir  : {TRAIN_DIR}")
    print()

    if not args.skip_download:
        _stage1_download(args.rebuild)
    _stage2_build_databases(args.rebuild)
    _stage3_build_features(args.rebuild)

    print()
    print("Done.")
    print(f"  Tasks     : {TRAIN_TASKS_JSON}")
    print(f"  CSVs      : {TRAIN_TS_DATA}")
    print(f"  Databases : {TRAIN_DATABASES}")


if __name__ == "__main__":
    main()
