"""Project-relative path constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# A benchmark directory holds `tasks.json` and `ts_data/<task_id>.csv`. The main
# benchmark lives in `nlqtsbench/`; `$SONAR_BENCHMARK_DIR` points the loader, the
# index builder and the runner at another one in the same layout, such as
# `nlqtsbench/lite/`. `$SONAR_DATABASES_DIR` relocates the per-task databases.
BENCHMARK_DIR: Path = Path(os.environ.get("SONAR_BENCHMARK_DIR") or PROJECT_ROOT / "nlqtsbench").resolve()
TS_DATA_DIR:   Path = BENCHMARK_DIR / "ts_data"
TASKS_JSON:    Path = BENCHMARK_DIR / "tasks.json"

DATABASES_DIR: Path = Path(os.environ.get("SONAR_DATABASES_DIR") or PROJECT_ROOT / "databases").resolve()

CONFIGS_DIR:     Path = PROJECT_ROOT / "configs"
OFFLINE_CONFIG:  Path = CONFIGS_DIR / "offline.yaml"

SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"


def db_path_for(task_id: str) -> Path:
    return DATABASES_DIR / f"{task_id}.sqlite"


def csv_path_for(task_id: str) -> Path:
    return TS_DATA_DIR / f"{task_id}.csv"
