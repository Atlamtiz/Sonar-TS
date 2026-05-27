---
id: shape-identification
description: For "identify the time range of the {superlative} {shape} in channel X within window W" — use SAX-regex on daily_feature to identify which days contain the shape, then verify by scanning the entire search window's raw_data with a shape-specific detector (the shape can span multiple days inside the window).
---

# Shape Identification — Search-Then-Verify with SAX + regex

This is one of two flagship tasks for Sonar-TS. The framework's whole point is **symbolic search + exact verification**. Both halves matter.

## SAX in 60 seconds

`daily_feature.sax` is a 24-character string per (channel, day). Within each day the raw values are z-normalised and binned into 5 equiprobable bands with letters `a-e`:

a = lowest 20% (deepest negative z-score in the day) b = 20-40% c = 40-60% (the day's median band) d = 60-80% e = 80-100% (highest band)

So SAX captures the **relative shape of one day's intraday curve**, not absolute values. A flat day reads `cccccccccccccccccccccccc` (or near it). A day with a sharp afternoon spike reads `aaaaaaaaaaaaeeeeaaaaaaaa`. A day with morning fall + afternoon rise reads `eeeebbbbbbbbbbbbeeeeeeee`.

Use this fact: **regex on SAX = approximate shape filtering**. The paper's canonical example is `WHERE regexp_like(sax, '[ab]+.*[de]+.*[ab]+')` for "rise then fall" — start low, peak high, end low.

## Search-Then-Verify pipeline

1. **Search (SQL).** Query `daily_feature` with a SAX regex matching the target shape. The framework has `regexp` and `regexp_like` pre-registered on the SQLite connection — write `WHERE sax REGEXP '<pattern>'` directly.

2. **Verify (Python).** Pull the entire search-window slice from `raw_data` (NOT day-by-day — the shape can span day boundaries). Run a shape-specific exact detector. Score each candidate; pick the best.

3. **Return.** `[start_timestamp_str, end_timestamp_str]`.

## SAX-regex templates (use these as starting points; generalise)

| Shape (from question) | SAX regex | Why |
|---|---|---|
| `plateau (stable period)` | `(.)\1{5,}` | any letter sustained 6+ hours |
| `low plateau (bottom out)` | `[ab]{5,}` | low band sustained |
| `upward spike` | `[a-c]+[de][a-c]+` or `[a-c]+e+[a-c]+` | low, sharp peak, low |
| `deep valley` | `[c-e]+[ab][c-e]+` | high, sharp trough, high |
| `step ascent` | `[ab]{3,}.*[de]{3,}` | sustained low → sustained high |
| `step descent` | `[de]{3,}.*[ab]{3,}` | sustained high → sustained low |

The regex is **fuzzy** — it identifies which days carry the shape, but not its exact boundary. Verify nails the boundary.

## Verify step — scan the WHOLE search window, not single days

Key insight: the search window often spans 2-10 days, and the requested shape may extend across day boundaries (e.g. a step ascent that begins in the morning of day 1 and finishes the afternoon of day 2). DO NOT restrict the detector to a single day at a time.

### plateau / depression — longest flat run
```python
# Pull the whole search window.
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
    conn, params=(SEARCH_START, SEARCH_END))
df["timestamp"] = pd.to_datetime(df["timestamp"])
v = df[CH].astype(float).values

# Local std as flatness indicator.
w = 4    # 1h smoothing
rstd = pd.Series(v).rolling(w, center=True).std().bfill().ffill().values
flat_mask = rstd <= 0.10 * float(np.std(v))
if SHAPE == "depression":
    flat_mask &= (v <= np.percentile(v, 30))

# Longest contiguous True run gives the plateau interval.
```

### spike / valley — single extremum amplitude
```python
# Index of the global extremum within search window.
idx = int(np.argmax(v)) if SHAPE == "spike" else int(np.argmin(v))
midline = ((np.max(v) + np.median(v)) / 2 if SHAPE == "spike"
           else (np.min(v) + np.median(v)) / 2)

# Walk outward from idx until value re-crosses the midline.
a = idx
while a > 0 and ((v[a-1] >= midline) if SHAPE == "spike" else (v[a-1] <= midline)):
    a -= 1
b = idx
while b < len(v)-1 and ((v[b+1] >= midline) if SHAPE == "spike" else (v[b+1] <= midline)):
    b += 1
```

### step_up / step_down — maximum-amplitude transition (cross-day OK)
```python
# Smooth aggressively (~3h) so micro-noise doesn't fragment the run.
W = max(8, int(round(pd.Timedelta(hours=3) /
                     df["timestamp"].diff().median())))
smooth = pd.Series(v).rolling(W, center=True).mean().bfill().ffill().values

# Find the pair of indices (a, b) with a < b that MAXIMISES (smooth[b] - smooth[a])
# for step_up, or minimises it for step_down. Vectorised in O(n):
if SHAPE == "step_up":
    # running min up to each position; gain = smooth[b] - running_min[b]
    run_min = np.minimum.accumulate(smooth)
    gain = smooth - run_min
    b = int(np.argmax(gain))
    a = int(np.argmin(smooth[:b + 1])) if b > 0 else 0
else:  # step_down
    run_max = np.maximum.accumulate(smooth)
    drop = run_max - smooth
    b = int(np.argmax(drop))
    a = int(np.argmax(smooth[:b + 1])) if b > 0 else 0
```

The O(n) running-min / running-max trick is **critical**: it finds the globally largest monotone transition, even if the underlying signal has small oscillations along the way. Day-by-day detectors miss cross-day ascents.

## End-to-end skeleton

```python
import re

CH = "<channel from question>"
SHAPE = "<plateau | depression | spike | valley | step_up | step_down>"
SUPERLATIVE = "<longest | highest | deepest | largest>"
SEARCH_START = "<YYYY-MM-DD HH:MM:SS>"
SEARCH_END   = "<YYYY-MM-DD HH:MM:SS>"

REGEX_BY_SHAPE = {
    "plateau":    r"(.)\1{5,}",
    "depression": r"[ab]{5,}",
    "spike":      r"[a-c]+[de][a-c]+",
    "valley":     r"[c-e]+[ab][c-e]+",
    "step_up":    r"[ab]{3,}.*[de]{3,}",
    "step_down":  r"[de]{3,}.*[ab]{3,}",
}

# 1. SQL search: do any days in the search window carry the shape?
rows = pd.read_sql_query(
    """SELECT window_start, window_end FROM daily_feature
       WHERE channel_id = ? AND window_start <= ? AND window_end >= ?
             AND sax REGEXP ?""",
    conn, params=(CH, SEARCH_END, SEARCH_START, REGEX_BY_SHAPE[SHAPE]),
)

# (We DON'T use rows to slice — they only confirm that the shape lives
# inside the search window. The verifier scans the entire window.)

# 2. Pull the WHOLE search window raw data once.
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
    conn, params=(SEARCH_START, SEARCH_END))
df["timestamp"] = pd.to_datetime(df["timestamp"])
v = df[CH].astype(float).values

# 3. Run the shape-specific detector to get (a, b).
# ...use the sketches above based on SHAPE...

# 4. Output timestamps.
_result = [df["timestamp"].iloc[a].strftime("%Y-%m-%d %H:%M:%S"),
           df["timestamp"].iloc[b].strftime("%Y-%m-%d %H:%M:%S")]
```

## Rules

* Use SAX regex to **confirm** the shape exists in the window (sanity check) but **always scan the entire search window**, not per-day.
* Use `regexp_like(sax, '...')` or `sax REGEXP '...'` — both work.
* Output interval is **typically 1-30 hours wide** for transient shapes (spike/valley); **6+ hours** for plateau/step.
* Never call `exit()` / `sys.exit()`.
