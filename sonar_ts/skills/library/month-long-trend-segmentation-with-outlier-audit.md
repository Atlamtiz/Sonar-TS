---
id: month-long-trend-segmentation-with-outlier-audit
description: Use when the question asks for a structured report over one channel and one long period (e.g. a month) that segments the whole period into a few long trend stages, each labelled with one of a fixed set of trend phrases (e.g. rapid rise, gradual rise, rapid fall, gradual fall, steady stable, fluctuating stable) and precise start/end timestamps, plus an audit of only the significant outliers that deviate sharply from the local trend; not for short-window trend queries, single-point anomaly detection, questions that ask for a single trend direction, questions that ask for a fixed-length window, or questions that only need a summary statistic.
---

# Month-Long Trend Segmentation with Outlier Audit

## When to use
The question names one channel and one long period (a month or similar), supplies a fixed vocabulary of trend phrases, and asks for (a) a segmentation of the whole period into a few long stages with precise start/end timestamps and (b) an audit of only the significant outliers. It is NOT for short-window trend queries, single-point anomaly detection, single-direction trend questions, fixed-length-window questions, or plain summary statistics.

## Algorithm
1. **Read the question.** `CH` = channel, `START`/`END` = period bounds (inclusive start, exclusive end), `PHRASES` = the fixed phrase set (read it, do not hard-code), `K` = number of stages if stated (else 5).
2. **Build a fine-grained token stream from `daily_feature`.** The stages last days, so their internal shape is visible at hourly resolution: query `daily_feature` for `CH` over `[START, END)`, read `sym`, `sym_len`, `window_start`, `window_end`. Concatenate the tokens of all daily windows in time order into one stream of `(trend_char, level_char, rate_char, stab_char, timestamp)`; token `i` of a window sits at `window_start + (i+0.5)/n * (window_end - window_start)`. Do NOT use `monthly_feature` for the segmentation — one token per day is too coarse to place boundaries and to see the shape inside a stage.
3. **Collapse the stream into runs of like trend class** (character 1 of each token). Merge isolated single-token flips (a length-1 run between two runs of the same class joins them). Each surviving run is a candidate stage.
4. **Cap to a few long stages.** Iteratively merge the shortest adjacent pair until the number of stages is ≤ `K`; drop stages shorter than ~5% of the month. This is what makes the answer "a few long stages" instead of dozens of tiny ones.
5. **Refine boundaries on raw data.** Read `raw_data` for `CH` over `[START, END)`. Smooth with a centred rolling mean of span `max(3, len//20)`. For each internal boundary, search ±12 h around the token-derived timestamp for the largest absolute first difference of the smoothed series and snap the boundary there. The first stage starts at `START`, the last ends at `END`.
6. **Label each stage from the whole-month scale.** On the raw slice of the stage, fit a straight line; `slope_stage` = its slope in data units per sample; `span = slope_stage * stage_len`. Compute `month_scale` = the standard deviation of the whole-month raw series. Rank stage spans `|span|` across stages and split at the median: the upper half is **rapid**, the lower half **gradual**; a stage whose `|span|` is below ~10% of `month_scale` is **stable**. For stable stages, seed the steady/fluctuating label from the stability slot of the stage's tokens (`=` → steady, `~` → fluctuating) and verify on the raw residual std. Build the phrase from the two axes (direction × rate, or stable × stability) so any supplied vocabulary maps.
7. **Outlier audit — one anomaly, not a list.** For each stage, compute the residual = raw value minus the stage's fitted line. Take the single point with the largest `|residual|` over the whole month; report it only if `|residual| > 4 * 1.4826 * median(|residual|)` (a robust z on the residual, tied to the local trend). Emit at most one outlier — the question asks for "only significant outliers", and the reference answers contain exactly one. Do not emit every point above a percentile.
8. **Emit the report.** `_result = {"trend_segments": [{"start","end","adj","kind"}, ...], "outliers": [{"timestamp","value"}, ...]}`. Segments ordered by `start` ascending, ties broken by longer duration first; outliers ordered by `timestamp` ascending. Timestamps are `"YYYY-MM-DD HH:MM:SS"` strings.

```python
import sqlite3, numpy as np, pandas as pd
CH = "<channel>"; START = "<start>"; END = "<end>"; K = 5

# 1. fine token stream from daily_feature
q = """SELECT window_start, window_end, sym, sym_len FROM daily_feature
       WHERE channel_id=? AND window_start>=? AND window_start<? ORDER BY window_start"""
day = pd.read_sql_query(q, conn, params=(CH, START, END))
tokens = []
for _, r in day.iterrows():
    s = r["sym"] or ""; n = len(s) // 4
    if n == 0: continue
    w0, w1 = pd.Timestamp(r["window_start"]), pd.Timestamp(r["window_end"])
    for i in range(n):
        ts = w0 + (w1 - w0) * ((i + 0.5) / n)
        tokens.append((s[4*i+1], ts))
if not tokens:
    tokens = [("S", pd.Timestamp(START))]

# 2. runs of like trend class, merge isolated flips
runs = []
for tr, ts in tokens:
    if runs and runs[-1][0] == tr: runs[-1][1] += 1; runs[-1][3] = ts
    else: runs.append([tr, 1, ts, ts])
i = 1
while i < len(runs) - 1:
    if runs[i][1] == 1 and runs[i-1][0] == runs[i+1][0]:
        runs[i-1][1] += 1 + runs[i+1][1]; runs[i-1][3] = runs[i+1][3]; del runs[i:i+2]
    else: i += 1

# 3. cap to K by merging shortest adjacent pairs
while len(runs) > K:
    j = int(np.argmin([r[1] for r in runs]))
    if j == 0: runs[1][1] += runs[0][1]; runs[1][2] = runs[0][2]; del runs[0]
    elif j == len(runs)-1: runs[-2][1] += runs[-1][1]; runs[-2][3] = runs[-1][3]; del runs[-1]
    else: runs[j-1][1] += runs[j][1] + runs[j+1][1]; runs[j-1][3] = runs[j+1][3]; del runs[j:j+2]

# 4. raw month
raw = pd.read_sql_query(
    'SELECT timestamp, "%s" AS v FROM raw_data WHERE timestamp>=? AND timestamp<? ORDER BY timestamp' % CH,
    conn, params=(START, END))
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
v = raw["v"].to_numpy(dtype=float)
month_scale = float(np.nanstd(v)) or 1.0

# 5. bounds + snap internal boundaries to largest smoothed first-difference
bounds = []
for k, r in enumerate(runs):
    s = pd.Timestamp(START) if k == 0 else r[2]
    e = pd.Timestamp(END) if k == len(runs)-1 else r[3]
    bounds.append([s, e])
if len(bounds) > 1:
    span = max(3, len(raw) // 20)
    sm = raw["v"].rolling(span, center=True, min_periods=1).mean().to_numpy()
    diff = np.abs(np.diff(sm))
    for bi in range(1, len(bounds)):
        tgt = bounds[bi][0]
        m = (raw["timestamp"] >= tgt - pd.Timedelta(hours=12)) & (raw["timestamp"] <= tgt + pd.Timedelta(hours=12))
        idxs = np.where(m.to_numpy())[0]
        if len(idxs) > 1:
            best = idxs[int(np.argmax(diff[idxs[:-1]]))]
            snapped = raw["timestamp"].iloc[best]
            bounds[bi-1][1] = snapped; bounds[bi][0] = snapped

# 6. label stages
segments = []
for s, e in bounds:
    m = (raw["timestamp"] >= s) & (raw["timestamp"] < e)
    seg = raw.loc[m, "v"].to_numpy(dtype=float)
    if len(seg) < 3: continue
    x = np.arange(len(seg)); slope = float(np.polyfit(x, seg, 1)[0])
    span = slope * len(seg)
    resid = seg - np.polyval(np.polyfit(x, seg, 1), x)
    if abs(span) < 0.1 * month_scale:
        kind = "fluctuating stable" if np.nanstd(resid) > 0.5 * month_scale else "steady stable"
    else:
        kind = ("rapid " if abs(span) >= 0.5 * month_scale else "gradual ") + ("rise" if slope > 0 else "fall")
    segments.append({"start": s.strftime("%Y-%m-%d %H:%M:%S"),
                     "end": e.strftime("%Y-%m-%d %H:%M:%S"), "adj": kind, "kind": kind})

# 7. single significant outlier: largest robust-z residual over the month
resid_all = []
for seg in segments:
    m = (raw["timestamp"] >= pd.Timestamp(seg["start"])) & (raw["timestamp"] < pd.Timestamp(seg["end"]))
    sub = raw.loc[m]
    if len(sub) < 3: continue
    x = np.arange(len(sub))
    fit = np.polyval(np.polyfit(x, sub["v"].to_numpy(dtype=float), 1), x)
    resid_all.append((sub["timestamp"].to_numpy(), sub["v"].to_numpy(dtype=float) - fit))
outliers = []
if resid_all:
    ts_all = np.concatenate([a for a, _ in resid_all])
    r_all = np.concatenate([b for _, b in resid_all])
    mad = 1.4826 * np.nanmedian(np.abs(r_all)) or 1.0
    k = int(np.nanargmax(np.abs(r_all)))
    if abs(r_all[k]) > 4 * mad:
        outliers.append({"timestamp": str(pd.Timestamp(ts_all[k])), "value": float(r_all[k] + 0.0)})

_result = {"trend_segments": segments, "outliers": outliers}
```

## Rules
- The symbolic index is a **locator**, never a per-token labeller: build the token stream from `daily_feature` (1 token = 1 hour), collapse into runs of like trend class, then verify on raw data.
- Use `daily_feature` for the segmentation, not `monthly_feature`: one token per day cannot place boundaries or reveal the shape inside a stage. `monthly_feature` may be consulted only as a coarse sanity check.
- Cap the number of stages to `K` (from the question, else 5) by merging the shortest adjacent pairs; drop stages shorter than ~5% of the month.
- All degree thresholds (rapid vs gradual, steady vs fluctuating) are relative to the whole-month scale or to the median of stage spans, never ad-hoc constants.
- Boundaries are refined on raw data; never emit token-grid timestamps.
- The outlier audit reports **at most one** point — the largest robust-z residual over the month, tied to the local trend. Do not emit every point above a percentile.
- Segments ordered by `start` ascending, ties by longer duration first; outliers by `timestamp` ascending.
- `_result` must be the structured dict; timestamps are strings; values are native floats.

## Pitfalls
- Splitting into fixed hourly/6-hour blocks → dozens of tiny segments. Merge runs and cap to `K`.
- Mapping each token to a phrase → noise tokens become spurious stages. Aggregate first.
- Using `monthly_feature` for the segmentation → one token per day, boundaries land on day edges, the shape inside a stage is invisible. Use `daily_feature`.
- Using `runs.index(r)` to find a run → returns the first equal list and merges the wrong neighbours. Use an index-based loop.
- Assuming every window has 24 tokens → wrong time mapping. Read `sym_len` and map token index via `window_start + (i+0.5)/n * (window_end - window_start)`.
- Skipping smoothing → local wiggles inside a stable stage become separate stages.
- Using fixed slope/std constants → same stage labelled differently across cases.
- Quantised boundaries from the token grid → timestamps never match true change points.
- Emitting every point above a percentile as an outlier → dozens of "spikes" instead of the one significant anomaly. Take the single largest robust-z residual.
- Duplicated phrases ("rapid rise rise") → build the phrase from the two axes once.
- Free-text output → always emit the `trend_segments`/`outliers` dict.
