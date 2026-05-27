"""Subprocess entry point for executing LLM-generated user code against a task DB."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
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

    conn = sqlite3.connect(db_path)
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


if __name__ == "__main__":
    _main()
