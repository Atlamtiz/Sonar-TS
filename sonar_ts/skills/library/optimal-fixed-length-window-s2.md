---
id: optimal-fixed-length-window-s2
description: Use when the question asks for the single fixed-length time window (a given number of days) inside a date range that optimises an aggregate of one channel — lowest average, highest average, or largest range — and requires the exact start and end timestamps of that window; not for variable-length extreme episodes, trend segmentation, joint upstream/downstream anomalies, or a single aggregate over a fixed date range with no sliding window.
---

# Sliding fixed-length window over raw timestamps

## When to use
The question names one channel, a date range, a window length in days, and an
objective: lowest average, highest average, or largest range. The answer is the
exact `[start, end]` timestamps of the single best window. It is NOT for
variable-length episodes (longest run above a threshold), for trend
segmentation, or for shape/form questions — those use the symbolic index.
Feature tables are not used here: the optimum is offset by hours, so
daily/monthly aggregates cannot locate it.

## Algorithm
1. Read from the question: `<channel>`, `<start>`, `<end>`, `<n_days>`, and the
   objective. Map the question's wording to `OBJ` explicitly:
   "lowest/smallest average/mean" → `min_mean`; "highest/largest average/mean"
   → `max_mean`; "largest/widest range/spread" → `max_range`. Reject any other
   wording rather than silently defaulting.
2. Pull the raw series once with a half-open range EXTENDED by the window length
   on the right, so windows starting near `<end>` are still evaluated:
   `WHERE timestamp >= <start> AND timestamp < <end> + <n_days> days`.
   Order by timestamp, convert to datetime, set as index.
3. Do NOT resample to calendar days and do NOT snap to day boundaries. The
   window is an exact time span of `<n_days>` days measured on the actual
   timestamps. Use a time-based rolling window:
   `roll = s.rolling(pd.Timedelta(days=<n_days>), closed='both')`.
   Only if the series is regular — modal step from
   `s.index.to_series().diff().mode()` and max deviation within tolerance — may
   you instead use `window = <n_days> * steps_per_day` rows.
4. Compute the objective on the rolling window:
   - mean objectives: `roll.mean()`
   - range objective: `roll.max() - roll.min()`
   Set `min_periods` explicitly to the expected count of samples in a full
   window (e.g. `int(<n_days> * steps_per_day * 0.95)`), so partial windows at
   the edges are excluded. NaNs are dropped before rolling; state this and keep
   it consistent between ranking and reporting.
5. Rank candidates by `(score, start)` — sort explicitly so ties break by the
   earliest start, not by index order. The window is half-open
   `[start, start + <n_days> days)`: `start = best_end - <n_days> days`, and
   `end` is the last sample strictly inside the window (the timestamp of the
   last row in the rolling window), not `start + N days`. Emit both as
   `"YYYY-MM-DD HH:MM:SS"` strings. Windows must start at or after `<start>`;
   clip candidates whose start falls before `<start>`.

```python
import pandas as pd, numpy as np

CH     = "<channel>"
START  = "<start>"
END    = "<end>"
N_DAYS = <n_days>
OBJ    = "<min_mean|max_mean|max_range>"

end_ext = (pd.to_datetime(END) + pd.Timedelta(days=N_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    'WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp',
    conn, params=(START, end_ext)
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna(subset=[CH]).set_index("timestamp").sort_index()
s = df[CH]

span = pd.Timedelta(days=N_DAYS)
# expected samples per full window (regular series); fall back to 1 if unknown
step = s.index.to_series().diff().mode()
steps_per_day = int(pd.Timedelta(days=1) / step.iloc[0]) if len(step) else 1
min_periods = max(1, int(N_DAYS * steps_per_day * 0.95))

roll = s.rolling(span, closed="both", min_periods=min_periods)
if OBJ == "max_range":
    score = roll.max() - roll.min()
elif OBJ == "max_mean":
    score = roll.mean()
elif OBJ == "min_mean":
    score = roll.mean()
else:
    raise ValueError(f"unknown objective: {OBJ}")

cand = pd.DataFrame({"score": score}).dropna()
cand["start"] = cand.index - span
cand = cand[cand["start"] >= pd.to_datetime(START)]
if OBJ == "min_mean":
    cand = cand.sort_values(["score", "start"], ascending=[True, True])
else:
    cand = cand.sort_values(["score", "start"], ascending=[False, True])
best = cand.iloc[0]

best_start = best["start"]
best_end   = best.name  # last sample inside the half-open window
_result = [best_start.strftime("%Y-%m-%d %H:%M:%S"),
           best_end.strftime("%Y-%m-%d %H:%M:%S")]
```

## Rules
- The window length is an exact time span (`<n_days>` days), never a count of
  calendar days or of rows.
- The search range is extended by `<n_days>` on the right so late-starting
  windows are considered; candidates must start at or after `<start>`.
- The interval is half-open `[start, start + <n_days> days)`; the end timestamp
  is the last sample inside the window, not `start + N days` inclusive.
- Rank on the exact aggregate of the channel over the full span, computed on
  raw_data — never on daily_feature/monthly_feature aggregates.
- Set `min_periods` explicitly; do not rely on the default of 1.
- Output exact timestamps, not whole-day spans (no forced `00:00:00` /
  `23:59:59`).
- Ties: earliest start wins — sort by `(score, start)` explicitly.

## Pitfalls
- Snapping to calendar days or using `daily_feature` rows as candidates — the
  true optimum is offset by hours.
- Using `rolling(N_DAYS)` on a resampled daily series — that is a row count, not
  a time span.
- Forgetting to extend the query range, so windows ending near `<end>` are
  missed.
- Emitting `start + N days` as the end (inclusive) instead of the last sample
  inside the half-open window.
- Relying on `idxmax`/`idxmin` for tie-breaking — it returns the first index
  occurrence, not the earliest start; sort explicitly.
- Silently defaulting an unrecognised objective to `min_mean`; map the wording
  and raise on anything else.
