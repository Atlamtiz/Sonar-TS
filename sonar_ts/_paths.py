"""Project-relative path constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

BENCHMARK_DIR: Path = PROJECT_ROOT / "nlqtsbench"
TS_DATA_DIR:   Path = BENCHMARK_DIR / "ts_data"
TASKS_JSON:    Path = BENCHMARK_DIR / "tasks.json"

DATABASES_DIR: Path = PROJECT_ROOT / "databases"

CONFIGS_DIR:     Path = PROJECT_ROOT / "configs"
OFFLINE_CONFIG:  Path = CONFIGS_DIR / "offline.yaml"

SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"


def db_path_for(task_id: str) -> Path:
    return DATABASES_DIR / f"{task_id}.sqlite"


def csv_path_for(task_id: str) -> Path:
    return TS_DATA_DIR / f"{task_id}.csv"
