"""Per-task PNG visualization, routed by subtask."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as _pio
from plotly.subplots import make_subplots

from ..storage import TaskDatabase

# Skip MathJax preloading; kaleido is optional so this is best-effort.
try:
    _pio.kaleido.scope.mathjax = None
except Exception:  # noqa: BLE001
    pass


_DOWNSAMPLE_TARGET = 1000
_COLOR_PRIMARY   = "#1f77b4"
_COLOR_BASELINE  = "#ff7f0e"
_COLOR_SECONDARY = "#2ca02c"
_COLOR_HIGHLIGHT = "#d62728"
_COLOR_ANNOT_BG  = "rgba(255,255,255,0.85)"

_TREND_FILL = {
    "rise":   "rgba(46, 125, 50, 0.20)",
    "fall":   "rgba(229, 57, 53, 0.20)",
    "stable": "rgba(96, 125, 139, 0.20)",
}


def _load_raw(task_id: str) -> pd.DataFrame:
    db = TaskDatabase(task_id)
    try:
        return db.fetch_raw()
    finally:
        db.close()


def _downsample(df: pd.DataFrame, target: int = _DOWNSAMPLE_TARGET) -> pd.DataFrame:
    if len(df) <= target:
        return df
    step = max(1, len(df) // target)
    return df.iloc[::step].reset_index(drop=True)


def _to_ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts
    except (ValueError, TypeError):
        return None


def _ts_str(value: Any) -> Optional[str]:
    ts = _to_ts(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts is not None else None


def _x_str(timestamps: pd.Series) -> List[str]:
    return timestamps.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()


def _layout(fig: go.Figure, title: str, height: int = 420, width: int = 1200) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        showlegend=True,
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        height=height, width=width,
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")


def _save(fig: go.Figure, out_path: Path, *, width: int = 1200, height: int = 420) -> None:
    fig.write_image(str(out_path), format="png", width=width, height=height)


def _placeholder(out_path: Path, title: str, message: str) -> None:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=14),
    )
    _layout(fig, title, height=300, width=900)
    _save(fig, out_path, width=900, height=300)


def _channel_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "timestamp"]


def _draw_line(fig: go.Figure, df: pd.DataFrame, ch: str, *,
               row: Optional[int] = None, col: Optional[int] = None,
               color: str = _COLOR_PRIMARY, name: Optional[str] = None,
               width: float = 1.2) -> None:
    sub = _downsample(df)
    trace = go.Scatter(
        x=_x_str(sub["timestamp"]), y=sub[ch].tolist(), mode="lines",
        name=name or ch, line=dict(color=color, width=width),
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _parse_interval(value: Any) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        a, b = _to_ts(value[0]), _to_ts(value[1])
        if a is not None and b is not None:
            return (a, b) if a <= b else (b, a)
    if isinstance(value, str):
        matches = re.findall(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?", value)
        if len(matches) >= 2:
            a, b = _to_ts(matches[0]), _to_ts(matches[1])
            if a is not None and b is not None:
                return (a, b) if a <= b else (b, a)
    return None


def _shade_interval(fig: go.Figure, a: pd.Timestamp, b: pd.Timestamp,
                    *, color: str = _COLOR_HIGHLIGHT, opacity: float = 0.35,
                    row: Optional[int] = None, col: Optional[int] = None,
                    line_color: str = "#b71c1c") -> None:
    kwargs = dict(
        x0=a.strftime("%Y-%m-%d %H:%M:%S"),
        x1=b.strftime("%Y-%m-%d %H:%M:%S"),
        fillcolor=color, opacity=opacity, layer="below",
        line=dict(color=line_color, width=1),
    )
    if row is not None and col is not None:
        fig.add_vrect(row=row, col=col, **kwargs)
    else:
        fig.add_vrect(**kwargs)


def _render_scalar(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    fig = go.Figure()
    _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)
    val_str = "(none)"
    if result is not None:
        try:
            val = float(result) if not isinstance(result, (list, tuple)) else float(result[0])
            val_str = f"{val:g}"
            fig.add_hline(y=val, line_dash="dash", line_color=_COLOR_HIGHLIGHT,
                          line_width=2)
        except (TypeError, ValueError):
            val_str = str(result)[:40]
    fig.add_annotation(
        text=f"Prediction: <b>{val_str}</b><br>{task.get('question', '')[:100]}",
        xref="paper", yref="paper", x=0.99, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False, bgcolor=_COLOR_ANNOT_BG, bordercolor="#444",
        borderwidth=1, font=dict(size=11),
    )
    _layout(fig, f"{task['id']}  ·  Global Aggregation")
    _save(fig, out_path)


def _render_timestamp(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    fig = go.Figure()
    _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)
    ts = _to_ts(result if not isinstance(result, (list, tuple)) else (result[0] if result else None))
    if ts is not None:
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        fig.add_vline(x=ts_str, line_color=_COLOR_HIGHLIGHT, line_width=2)
        fig.add_annotation(
            x=ts_str, y=1.0, yref="paper",
            text=f"  {ts_str}",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(color=_COLOR_HIGHLIGHT, size=11),
            bgcolor="rgba(255,255,255,0.85)",
        )
    _layout(fig, f"{task['id']}  ·  Temporal Localization")
    _save(fig, out_path)


def _render_interval(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    fig = go.Figure()
    _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)
    interval = _parse_interval(result)
    if interval is not None:
        _shade_interval(fig, *interval)
        fig.add_annotation(
            text=f"Predicted interval:<br>{interval[0]} → {interval[1]}",
            xref="paper", yref="paper", x=0.99, y=0.98,
            xanchor="right", yanchor="top",
            showarrow=False, bgcolor=_COLOR_ANNOT_BG, bordercolor="#444",
            borderwidth=1, font=dict(size=10),
        )
    _layout(fig, f"{task['id']}  ·  {task.get('subtask', '')}")
    _save(fig, out_path)


def _render_shape(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    q = task.get("question", "")
    matches = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", q)
    if len(matches) >= 2:
        a, b = _to_ts(matches[0]), _to_ts(matches[1])
        if a is not None and b is not None:
            sub = df[(df["timestamp"] >= a) & (df["timestamp"] <= b)].reset_index(drop=True)
            if not sub.empty:
                df = sub
    fig = go.Figure()
    _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)
    interval = _parse_interval(result)
    if interval is not None:
        _shade_interval(fig, *interval)
    _layout(fig, f"{task['id']}  ·  Shape Identification")
    _save(fig, out_path)


def _render_period(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    q = task.get("question", "")
    matches = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", q)
    if len(matches) >= 2:
        a, b = _to_ts(matches[0]), _to_ts(matches[1])
        if a is not None and b is not None:
            sub = df[(df["timestamp"] >= a) & (df["timestamp"] <= b)].reset_index(drop=True)
            if not sub.empty:
                df = sub
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    fig = go.Figure()
    _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)
    period_val = None
    if result is not None:
        try:
            period_val = int(float(result if not isinstance(result, (list, tuple)) else result[0]))
        except (TypeError, ValueError):
            period_val = None
    if period_val and period_val > 1 and len(df) > period_val:
        max_lines = min(20, len(df) // period_val)
        for k in range(1, max_lines + 1):
            i = k * period_val
            if i >= len(df):
                break
            fig.add_vline(x=df["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),
                          line_color="#e53935", line_width=1.5, line_dash="dash",
                          opacity=0.7)
        fig.add_annotation(
            text=f"Predicted period: <b>{period_val}</b> data points",
            xref="paper", yref="paper", x=0.99, y=0.98,
            xanchor="right", yanchor="top",
            showarrow=False, bgcolor=_COLOR_ANNOT_BG, bordercolor="#444",
            borderwidth=1, font=dict(size=11),
        )
    _layout(fig, f"{task['id']}  ·  Periodicity Detection")
    _save(fig, out_path)


def _render_subseq(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    q = task.get("question", "")
    matches = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", q)
    timestamps = [_to_ts(m) for m in matches]
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) < 4:
        return _render_interval(task, result, out_path)
    q_start, q_end, s_start, s_end = timestamps[:4]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Reference (query window)",
                                        "Search window  ·  shaded = prediction"))

    q_df = df[(df["timestamp"] >= q_start) & (df["timestamp"] <= q_end)].reset_index(drop=True)
    s_df = df[(df["timestamp"] >= s_start) & (df["timestamp"] <= s_end)].reset_index(drop=True)

    if not q_df.empty:
        fig.add_trace(go.Scatter(
            x=_x_str(q_df["timestamp"]), y=q_df[ch].tolist(), mode="lines",
            name="query", line=dict(color=_COLOR_BASELINE, width=1.5),
        ), row=1, col=1)
    if not s_df.empty:
        fig.add_trace(go.Scatter(
            x=_x_str(s_df["timestamp"]), y=s_df[ch].tolist(), mode="lines",
            name="search", line=dict(color=_COLOR_PRIMARY, width=1.2),
        ), row=1, col=2)

    interval = _parse_interval(result)
    if interval is not None:
        _shade_interval(fig, *interval, row=1, col=2)

    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    fig.update_layout(
        title=dict(text=f"{task['id']}  ·  Subsequence Matching", font=dict(size=12)),
        showlegend=True,
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, width=1400,
        font=dict(size=11),
    )
    _save(fig, out_path, width=1400, height=420)


def _render_composite_trend(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    dates: List[str] = []
    if isinstance(result, (list, tuple)):
        dates = [str(d)[:10] for d in result if d]
    elif isinstance(result, str):
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", result)
    dates = [d for d in dates if re.match(r"^\d{4}-\d{2}-\d{2}$", d)][:6]
    if not dates:
        _placeholder(out_path, task["id"], "(no dates in prediction)")
        return

    n = len(dates)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=dates,
                        shared_yaxes=False, vertical_spacing=0.12)
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    for i, d in enumerate(dates):
        r, c = i // cols + 1, i % cols + 1
        sub = df[df["date"] == pd.to_datetime(d).date()].reset_index(drop=True)
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=_x_str(sub["timestamp"]), y=sub[ch].tolist(), mode="lines",
            name=d, line=dict(color=_COLOR_PRIMARY, width=1.2),
            showlegend=False,
        ), row=r, col=c)

    fig.update_xaxes(showgrid=True, gridcolor="#eee", tickformat="%H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    fig.update_layout(
        title=dict(text=f"{task['id']}  ·  Composite Trend  ·  {task.get('question', '')[:120]}",
                   font=dict(size=11)),
        showlegend=False,
        margin=dict(l=50, r=20, t=70, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        height=300 * rows, width=420 * cols,
        font=dict(size=10),
    )
    _save(fig, out_path, width=420 * cols, height=300 * rows)


def _render_contextual_anomaly(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    hm_col = f"{ch}_hist_mean"
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    fig = go.Figure()
    _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)
    if hm_col in df.columns:
        sub = _downsample(df)
        fig.add_trace(go.Scatter(
            x=_x_str(sub["timestamp"]), y=sub[hm_col].tolist(), mode="lines",
            name="hist_mean (baseline)",
            line=dict(color=_COLOR_BASELINE, width=1.3, dash="dash"),
        ))
    interval = _parse_interval(result)
    if interval is not None:
        _shade_interval(fig, *interval)
    _layout(fig, f"{task['id']}  ·  Contextual Anomaly")
    _save(fig, out_path)


def _render_causal_anomaly(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return
    channels = _channel_columns(df)
    interval = _parse_interval(result)
    if interval is not None:
        a, b = interval
        width = b - a
        pad = width * 2 if width > pd.Timedelta(0) else pd.Timedelta(days=3)
        lo, hi = a - pad, b + pad
        sub = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(drop=True)
        if not sub.empty:
            df = sub
    fig = go.Figure()
    colors = [_COLOR_PRIMARY, _COLOR_BASELINE, _COLOR_SECONDARY]
    for i, c in enumerate(channels):
        _draw_line(fig, df, c, color=colors[i % len(colors)], name=c, width=1.3)
    if interval is not None:
        _shade_interval(fig, *interval)
    _layout(fig, f"{task['id']}  ·  Causal Anomaly  (focused view)")
    _save(fig, out_path)


_IS_SEG_COLORS = {
    ("gradual", "rise"):       "#e64a19",
    ("rapid", "rise"):         "#d81b60",
    ("gradual", "fall"):       "#00897b",
    ("rapid", "fall"):         "#1565c0",
    ("steady", "stable"):      "#546e7a",
    ("fluctuating", "stable"): "#8e24aa",
}
_IS_FALLBACK_PALETTE = ["#9c27b0", "#e64a19", "#1976d2", "#388e3c",
                        "#fbc02d", "#00897b", "#5d4037"]


def _render_insight_synthesis(task: Dict[str, Any], result: Any, out_path: Path) -> None:
    df = _load_raw(task["id"])
    ch = task["channel"]
    if df.empty:
        _placeholder(out_path, task["id"], "(no raw data)")
        return

    title_month = ""
    m = re.search(r"(\d{4})-(\d{2})", task.get("question", ""))
    if m:
        Y, M = int(m.group(1)), int(m.group(2))
        title_month = f"{Y}-{M:02d}"
        start = pd.Timestamp(f"{Y}-{M:02d}-01")
        end = start + pd.offsets.MonthBegin(1)
        clipped = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
        if not clipped.empty:
            df = clipped

    fig = go.Figure()
    df_dense = _downsample(df)

    if isinstance(result, dict) and (result.get("trend_segments") or result.get("outliers")):
        fig.add_trace(go.Scatter(
            x=_x_str(df_dense["timestamp"]), y=df_dense[ch].tolist(),
            mode="lines", name="channel", line=dict(color="#cfd8dc", width=1.0),
            showlegend=False, hoverinfo="skip",
        ))

        for i, seg in enumerate(result.get("trend_segments", []) or []):
            if not isinstance(seg, dict):
                continue
            a, b = _to_ts(seg.get("start")), _to_ts(seg.get("end"))
            if a is None or b is None:
                continue
            adj = str(seg.get("adj", "")).lower().strip()
            kind = str(seg.get("kind", "")).lower().strip()
            color = (_IS_SEG_COLORS.get((adj, kind))
                     or _IS_FALLBACK_PALETTE[i % len(_IS_FALLBACK_PALETTE)])
            seg_df = df[(df["timestamp"] >= a) & (df["timestamp"] <= b)].reset_index(drop=True)
            if seg_df.empty:
                continue
            seg_df = _downsample(seg_df, target=800)
            fig.add_trace(go.Scatter(
                x=_x_str(seg_df["timestamp"]), y=seg_df[ch].tolist(),
                mode="lines",
                name=(f"{adj} {kind}".strip() or f"segment {i + 1}"),
                line=dict(color=color, width=1.6),
            ))

        # Look up the actual channel value at the predicted timestamp;
        # the predict.json round-trip strips the numeric `value` field.
        for o in result.get("outliers", []) or []:
            if not isinstance(o, dict):
                continue
            ts = _to_ts(o.get("timestamp"))
            if ts is None:
                continue
            row = df[df["timestamp"] == ts]
            if row.empty:
                # nearest-neighbour fallback for sub-step timestamp mismatches
                deltas = (df["timestamp"] - ts).abs()
                if len(deltas):
                    row = df.iloc[[int(deltas.idxmin())]]
            if row.empty:
                val = o.get("value")
                if val is None:
                    continue
                val = float(val)
            else:
                val = float(row.iloc[0][ch])
            fig.add_trace(go.Scatter(
                x=[ts.strftime("%Y-%m-%d %H:%M:%S")], y=[float(val)],
                mode="markers", name="significant outlier",
                marker=dict(symbol="triangle-up", color=_COLOR_HIGHLIGHT,
                            size=14, line=dict(width=1.5, color="#7f0000")),
            ))
            fig.add_vline(x=ts.strftime("%Y-%m-%d %H:%M:%S"),
                          line_color=_COLOR_HIGHLIGHT, line_width=1,
                          line_dash="dash", opacity=0.5)
    else:
        _draw_line(fig, df, ch, color=_COLOR_PRIMARY, name=ch)

    title = f"{task['id']}  ·  Insight Synthesis"
    if title_month:
        title += f"  ·  {title_month}"
    _layout(fig, title)
    _save(fig, out_path)


def _render_placeholder(task: Dict[str, Any], result: Any, out_path: Path,
                        msg: str = "no visualization for this subtask") -> None:
    _placeholder(out_path, task.get("id", "<no id>"),
                 f"{task.get('subtask', '?')} — {msg}")


_DISPATCH = {
    "Global Aggregation":     _render_scalar,
    "Temporal Localization":  _render_timestamp,
    "Interval Discovery":     _render_interval,
    "Sliding Window":         _render_interval,
    "Shape Identification":   _render_shape,
    "Periodicity Detection":  _render_period,
    "Subsequence Matching":   _render_subseq,
    "Composite Trend":        _render_composite_trend,
    "Contextual Anomaly":     _render_contextual_anomaly,
    "Causal Anomaly":         _render_causal_anomaly,
    "Insight Synthesis":      _render_insight_synthesis,
}


def save_figure(task: Dict[str, Any], result: Any, out_path: Path | str,
                **_kwargs) -> Path:
    """Render the appropriate figure for ``task``'s subtask; always emits a PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subtask = task.get("subtask", "")
    renderer = _DISPATCH.get(subtask, _render_placeholder)
    try:
        renderer(task, result, out_path)
    except Exception as exc:  # noqa: BLE001
        _render_placeholder(task, result, out_path,
                            msg=f"render error: {type(exc).__name__}: {exc}")
    return out_path
