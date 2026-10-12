"""Subprocess entry point for executing LLM-generated user code against a task DB."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401


def _main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: _runner.py <db_path> <user_code_path>\n")
        sys.exit(2)
    db_path, code_path = sys.argv[1], sys.argv[2]

    with open(code_path, "r", encoding="utf-8") as f:
        user_code = f.read()

    # SQL time of the script (execute + fetch of every statement), reported on
    # stderr at the end as SONAR_SQL_SECONDS=<seconds>; the executor reads it.
    _sql_clock = [0.0]

    class _TimedCursor(sqlite3.Cursor):
        def _timed(self, fn, *a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                _sql_clock[0] += time.perf_counter() - t0

        def execute(self, *a, **k):
            return self._timed(super().execute, *a, **k)

        def executemany(self, *a, **k):
            return self._timed(super().executemany, *a, **k)

        def fetchall(self):
            return self._timed(super().fetchall)

        def fetchmany(self, *a, **k):
            return self._timed(super().fetchmany, *a, **k)

        def fetchone(self):
            return self._timed(super().fetchone)

    class _TimedConnection(sqlite3.Connection):
        def cursor(self, factory=_TimedCursor):
            return super().cursor(factory)

    conn = sqlite3.connect(db_path, factory=_TimedConnection)
    conn.row_factory = sqlite3.Row

    _re_cache: dict[str, "re.Pattern[str]"] = {}

    def _regexp(pattern: str, value: object) -> bool:
        if value is None:
            return False
        if pattern not in _re_cache:
            try:
                _re_cache[pattern] = re.compile(pattern)
            except re.error:
                return False
        return _re_cache[pattern].search(str(value)) is not None

    conn.create_function("regexp", 2, _regexp, deterministic=True)
    conn.create_function("regexp_like", 2,
                         lambda value, pattern: _regexp(pattern, value),
                         deterministic=True)

    globals_for_code = {
        "__name__": "__main__",
        "conn": conn,
        "pd": pd,
        "np": np,
        "sqlite3": sqlite3,
        "json": json,
        "math": math,
        "re": re,
        "datetime": datetime,
        "timedelta": timedelta,
    }

    try:
        exec(compile(user_code, code_path, "exec"), globals_for_code)
    finally:
        conn.close()
        sys.stderr.write(f"\nSONAR_SQL_SECONDS={_sql_clock[0]:.4f}\n")
        sys.stderr.flush()


if __name__ == "__main__":
    _main()
