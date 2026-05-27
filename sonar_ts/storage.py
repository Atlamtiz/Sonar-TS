"""Per-task SQLite storage."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import pandas as pd

from ._paths import db_path_for, csv_path_for


FEATURE_TABLES: tuple[str, ...] = ("yearly_feature", "monthly_feature", "daily_feature")
ALL_TABLES: tuple[str, ...] = ("raw_data",) + FEATURE_TABLES


_DDL_FEATURE_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {name} (
    channel_id    TEXT NOT NULL,
    window_start  TEXT NOT NULL,
    window_end    TEXT NOT NULL,
    min_val       REAL,
    max_val       REAL,
    avg_val       REAL,
    std_val       REAL,
    slope         REAL,
    sax_len       INTEGER,
    sax           TEXT,
    PRIMARY KEY (channel_id, window_start)
);
CREATE INDEX IF NOT EXISTS idx_{name}_ch  ON {name}(channel_id);
CREATE INDEX IF NOT EXISTS idx_{name}_sax ON {name}(sax);
"""


def _ddl_for_features() -> str:
    return "".join(_DDL_FEATURE_TEMPLATE.format(name=n) for n in FEATURE_TABLES)


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class TaskDatabase:
    def __init__(self, task_id: str, *, db_path: Path | str | None = None):
        self.task_id = task_id
        self.db_path = Path(db_path) if db_path is not None else db_path_for(task_id)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "TaskDatabase":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def init_feature_schema(self) -> None:
        conn = self.connect()
        conn.executescript(_ddl_for_features())
        conn.commit()

    def has_table(self, name: str) -> bool:
        cur = self.connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None

    def table_row_count(self, name: str) -> int:
        if not self.has_table(name):
            return 0
        cur = self.connect().execute(f"SELECT COUNT(*) FROM {name}")
        return int(cur.fetchone()[0])

    def table_columns(self, name: str) -> List[str]:
        cur = self.connect().execute(f"PRAGMA table_info({_quote_identifier(name)})")
        return [row[1] for row in cur.fetchall()]

    def raw_channel_columns(self) -> List[str]:
        return [c for c in self.table_columns("raw_data") if c != "timestamp"]

    @classmethod
    def from_csv(cls, task_id: str, csv_path: Path | str | None = None,
                 *, db_path: Path | str | None = None,
                 overwrite: bool = False) -> "TaskDatabase":
        csv_path = Path(csv_path) if csv_path is not None else csv_path_for(task_id)
        db = cls(task_id, db_path=db_path)
        if overwrite and db.db_path.exists():
            db.db_path.unlink()
        db.init_feature_schema()
        db._load_csv(csv_path)
        return db

    def _load_csv(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"source CSV missing: {csv_path}")
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{csv_path}: required 'timestamp' column not found")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        channel_cols = [c for c in df.columns if c != "timestamp"]
        if not channel_cols:
            raise ValueError(f"{csv_path}: no numeric channel columns")
        for col in channel_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        conn = self.connect()
        col_defs = ['"timestamp" TEXT PRIMARY KEY NOT NULL'] + [
            f'{_quote_identifier(c)} REAL' for c in channel_cols
        ]
        conn.execute("DROP TABLE IF EXISTS raw_data")
        conn.execute(f"CREATE TABLE raw_data ({', '.join(col_defs)})")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_data(timestamp)")

        df.to_sql("raw_data", conn, if_exists="append", index=False)
        conn.commit()

    def execute_sql(self, sql: str, params: Sequence = ()) -> list[dict]:
        cur = self.connect().execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def fetch_raw(self, *, channels: List[str] | None = None,
                  start: str | None = None,
                  end: str | None = None) -> pd.DataFrame:
        all_channels = self.raw_channel_columns()
        cols = channels if channels is not None else all_channels
        unknown = [c for c in cols if c not in all_channels]
        if unknown:
            raise ValueError(f"unknown channels {unknown}; available: {all_channels}")

        select = ", ".join(['"timestamp"'] + [_quote_identifier(c) for c in cols])
        where: list[str] = []
        params: list = []
        if start is not None:
            where.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            where.append("timestamp <= ?")
            params.append(end)
        sql = f"SELECT {select} FROM raw_data"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp"
        df = pd.read_sql_query(sql, self.connect(), params=params)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def fetch_raw_long(self) -> pd.DataFrame:
        wide = self.fetch_raw()
        if wide.empty:
            return pd.DataFrame(columns=["timestamp", "channel_id", "value"])
        long = wide.melt(id_vars="timestamp", var_name="channel_id", value_name="value")
        return long.dropna(subset=["value"]).reset_index(drop=True)

    def write_feature_table(self, name: str, df: pd.DataFrame) -> None:
        if name not in FEATURE_TABLES:
            raise ValueError(f"{name!r} is not a valid feature-table name")
        expected = {"channel_id", "window_start", "window_end",
                    "min_val", "max_val", "avg_val", "std_val",
                    "slope", "sax_len", "sax"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"{name}: DataFrame missing columns {sorted(missing)}")
        conn = self.connect()
        conn.execute(f"DELETE FROM {name}")
        df[list(expected)].to_sql(name, conn, if_exists="append", index=False)
        conn.commit()


@contextmanager
def open_task_db(task_id: str) -> Iterator[TaskDatabase]:
    db = TaskDatabase(task_id)
    try:
        db.connect()
        yield db
    finally:
        db.close()
