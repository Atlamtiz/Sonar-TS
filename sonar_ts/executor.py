"""Run generated Python in a subprocess and classify the outcome."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from ._paths import PROJECT_ROOT, db_path_for

_RUNNER = PROJECT_ROOT / "sonar_ts" / "_runner.py"


@dataclass
class ExecResult:
    status: str
    result: Any = None
    message: str = ""
    stdout: str = ""
    stderr: str = ""

    def is_ok(self) -> bool:
        return self.status == "ok"


def _looks_like_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        import pandas as pd
        pd.Timestamp(value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _validate_shape(result: Any, eval_metric: str,
                    shape_rules: Dict[str, Dict]) -> str | None:
    rule = shape_rules.get(eval_metric)
    if not rule:
        return None
    expected = rule.get("expected")

    if expected == "scalar":
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            return f"expected a scalar number, got {type(result).__name__}: {result!r}"
        return None

    if expected == "timestamp":
        if not _looks_like_timestamp(result):
            return f"expected a single timestamp string, got: {result!r}"
        return None

    if expected == "interval":
        if not (isinstance(result, list) and len(result) == 2):
            return f"expected a list of two timestamps, got: {result!r}"
        if not all(_looks_like_timestamp(x) for x in result):
            return f"interval elements are not parseable as timestamps: {result!r}"
        return None

    if expected == "date_set":
        if not isinstance(result, list):
            return f"expected a list of dates, got {type(result).__name__}"
        max_items = rule.get("max_items", 50)
        if len(result) > max_items:
            return f"date list length {len(result)} exceeds max {max_items}"
        if not all(isinstance(x, str) and len(x) >= 10 for x in result):
            return f"date list elements are not date strings: {result[:3]}..."
        return None

    if expected == "nonempty_string":
        if not isinstance(result, str) or not result.strip():
            return f"expected a non-empty string report, got: {result!r}"
        return None

    if expected == "report_dict":
        if isinstance(result, dict):
            if "trend_segments" not in result and "segments" not in result:
                return ("report dict missing 'trend_segments' (or 'segments') "
                        f"key; got keys: {list(result.keys())}")
            return None
        if isinstance(result, str) and result.strip():
            return None
        return (f"expected a dict {{'trend_segments': [...], 'outliers': [...]}} "
                f"or a non-empty string, got: {type(result).__name__}")

    return None


def execute(code: str, task_id: str, eval_metric: str,
            *, timeout_seconds: int, shape_rules: Dict[str, Dict]) -> ExecResult:
    db_path = db_path_for(task_id)
    if not db_path.exists():
        return ExecResult(status="exception",
                          message=f"database missing: {db_path}")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        tmp_path = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(_RUNNER), str(db_path), str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return ExecResult(
            status="timeout",
            message=(f"code exceeded {timeout_seconds}s — likely an O(N) Python "
                     "loop on raw_data; prefer SQL aggregation or feature-table "
                     "filtering before raw scans"),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().split("\n")[-30:])
        return ExecResult(
            status="exception",
            message=stderr_tail,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    stdout = proc.stdout.strip()
    last_line = stdout.split("\n")[-1] if stdout else ""
    try:
        parsed = json.loads(last_line)
    except json.JSONDecodeError:
        return ExecResult(
            status="format_invalid",
            message=(f"final stdout line is not valid JSON. "
                     f"Got (last 500 chars): {stdout[-500:]!r}"),
            stdout=proc.stdout,
        )
    if not isinstance(parsed, dict) or "_result" not in parsed:
        return ExecResult(
            status="format_invalid",
            message=(f"expected JSON with a '_result' key, got: {parsed!r}"),
            stdout=proc.stdout,
        )

    result_value = parsed["_result"]
    shape_err = _validate_shape(result_value, eval_metric, shape_rules)
    if shape_err is not None:
        return ExecResult(
            status="shape_mismatch",
            result=result_value,
            message=shape_err,
            stdout=proc.stdout,
        )

    return ExecResult(status="ok", result=result_value, stdout=proc.stdout)
