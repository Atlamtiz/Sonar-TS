"""Per-eval_metric format converters from native values to evaluator strings."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _to_ts_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            return pd.Timestamp(value).strftime(_TS_FMT)
        except (ValueError, TypeError):
            return value
    if isinstance(value, pd.Timestamp):
        return value.strftime(_TS_FMT)
    try:
        return pd.Timestamp(value).strftime(_TS_FMT)
    except (ValueError, TypeError):
        return str(value)


def _to_date_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        if len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return value[:10]
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return value
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)


def _trim_float(x: float, max_decimals: int = 6) -> str:
    s = f"{x:.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _trim_float(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return format_scalar(value[0])
    return str(value)


def format_timestamp(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return _to_ts_str(value)


def format_interval(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        start = _to_ts_str(value[0])
        end = _to_ts_str(value[1])
        return f"[{start}, {end}]"
    if isinstance(value, dict):
        start = _to_ts_str(value.get("start") or value.get("begin") or value.get("from"))
        end = _to_ts_str(value.get("end") or value.get("stop") or value.get("to"))
        if start and end:
            return f"[{start}, {end}]"
    return str(value)


def format_date_set(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("dates", "result", "answer", "top_k", "topk"):
            if k in value and isinstance(value[k], (list, tuple)):
                value = value[k]
                break
        else:
            return str(value)
    if isinstance(value, (list, tuple)):
        dates = [_to_date_str(v) for v in value]
        return "[" + ", ".join(f"'{d}'" for d in dates) + "]"
    return str(value)


def format_report(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    if not isinstance(value, dict):
        return str(value)

    segments = value.get("trend_segments") or value.get("segments") or []
    outliers = value.get("outliers") or value.get("anomalies") or []

    seg_parts = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        start = _to_ts_str(s.get("start"))
        end = _to_ts_str(s.get("end"))
        adj = str(s.get("adj") or s.get("adjective") or "").strip().lower()
        kind = str(s.get("kind") or s.get("type") or s.get("trend") or "").strip().lower()
        if start and end and adj and kind:
            seg_parts.append(
                f"from {start} to {end}, the trend showed a {adj} {kind}"
            )
    seg_text = ("; ".join(seg_parts) + ".") if seg_parts else "None."

    out_parts = []
    for o in outliers:
        if not isinstance(o, dict):
            continue
        ts = _to_ts_str(o.get("timestamp") or o.get("time") or o.get("at"))
        val = o.get("value")
        if not ts:
            continue
        if val is not None:
            try:
                out_parts.append(
                    f"A significant spike was detected at {ts} (value: {float(val):.2f})"
                )
                continue
            except (TypeError, ValueError):
                pass
        out_parts.append(f"A significant spike was detected at {ts}")
    out_text = (". ".join(out_parts) + ".") if out_parts else "None."

    return f"1. Trend Segmentation: {seg_text}\n2. Outlier Audit: {out_text}"


_DISPATCH = {
    "rel_acc": format_scalar,
    "hit":     format_timestamp,
    "iou":     format_interval,
    "set_f1":  format_date_set,
    "report":  format_report,
}


def format_prediction(task: Dict[str, Any], result: Any) -> str:
    if result is None:
        return ""
    fn = _DISPATCH.get(task.get("eval_metric", ""))
    if fn is None:
        return str(result)
    try:
        return fn(result)
    except Exception:  # noqa: BLE001
        return str(result)
