---
id: two-channel-causal-anomaly-interval
description: Use when the question names an upstream source channel and a downstream channel and asks for the time interval of a causal anomaly in the downstream series — an inverse trend or flat line while the upstream is active — where the interval must cover the full multi-day episode and the episode is ranked by cumulative divergence between the two series over the episode. Not for single-channel named shape features (plateau, spike, step, valley), for sustained extreme episodes scored by cumulative anomaly in one channel, or for global trend/seasonality summaries.
---

# Two-channel causal anomaly interval

## When to use
The question names an upstream channel and a downstream channel and asks for the time period where the downstream shows a causal anomaly — a flat line during high upstream activity, or an inverse trend against the source. The episode may span multiple days, so the interval must cover the whole run, not a single day. Not for single-channel shape questions and not for plain aggregates.

## Algorithm

1. **Read the question.** Extract `<up_ch>`, `<down_ch>`, `<start>`, `<end>`, and the anomaly kind `<flat|inverse>`. The episode length is not stated; derive the minimum run length from the token length of the chosen table (a run of at least 2 tokens is the smallest shape the index can express).

2. **Pick the feature table by the DURATION OF THE SHAPE, not the question's time span.** A shape lasting days → `monthly_feature` (1 token = 1 day); a shape lasting hours → `daily_feature`. Decide from the wording of the anomaly (a "flat line during high activity" over a multi-day episode is a days-scale shape), and if unsure, search `monthly_feature` first and fall back to `daily_feature`. Never assume a fixed table from the data.

3. **Search the index (loose first), on BOTH channels.**
   - Downstream flat: a run of stable tokens `(?:[a-e]S0[=~]){2,}`.
   - Downstream inverse: a run of falling tokens `(?:[a-e]F[012][=~])(?:[a-e][RFS][012][=~])*?(?:[a-e]F[012][=~])` (or rising, per the question) — never glue the falling tokens directly; insert the whole-token skip between parts.
   - Upstream active: the same table for `<up_ch>` over the same window range with `std_val` above the median of `<up_ch>`'s `std_val` over `[<start>, <end>)`, or `sym` containing `R`/`F` tokens.
   - **Filter by overlap, not by start:** `window_start < <end> AND window_end > <start>`, so an episode that begins in the last days of a month is not truncated.
   - **If the strict pattern returns fewer windows than needed, widen** the rate/stability slots to `[012]`/`[=~]`, then fall back to raw_data. Never return empty.

4. **Verify on raw_data over the shortlisted span, padded by one full window on each side.** Read both channels with one SQL query each. Compute the exact interval:
   - **Flat anomaly:** a **fixed band around the run's own median**, not a drifting running mean (a slow ramp must not read as flat). `band = 2 * median(|diff(v)|)`; if a single large jump inflates that, fall back to the IQR of diffs, or to the containing monthly window's `std_val`. Extend the run while `|v[i] - median(run)| <= band`. Require a minimum run length in samples (at least one full window's worth) so a 2-point coincidence cannot win.
   - **Inverse anomaly:** the longest run where the sign of the downstream slope (short rolling window) is opposite to the upstream slope, and the upstream is active.
   - **Upstream-active test (one definition, used everywhere):** upstream rolling std over the candidate span > median of upstream rolling std over the whole question range. State it once; do not mix in `sym` or `std_val` as alternative thresholds.
   - **Vectorise** the run search with `np.cumsum` / `pandas.rolling`; do not loop in Python over every sample of a year of sub-hourly data.

5. **Rank.** Rank by the **sustained co-occurrence** of upstream activity with downstream flatness/inversion over the whole episode — e.g. the length of the co-occurring run, or the product of upstream std and downstream flatness duration. Not an ad-hoc ratio. Ties → earliest start (make the tie-break explicit: keep the earlier run when lengths are equal).

6. **Return** `[start_str, end_str]` as `"YYYY-MM-DD HH:MM:SS"` strings. If no candidate survives, fall back to the widest candidate window from the index, or to the longest run found on the unpadded read — never return the whole question range.

```python
import pandas as pd, numpy as np

UP = "<up_ch>"; DOWN = "<down_ch>"
START = "<start>"; END = "<end>"
KIND = "<flat|inverse>"

# --- 1. index search: shape duration decides the table ---
if KIND == "flat":
    pat = r"(?:[a-e]S0[=~]){2,}"
else:
    pat = r"(?:[a-e]F[012][=~])(?:[a-e][RFS][012][=~])*?(?:[a-e]F[012][=~])"

def search(tbl):
    return pd.read_sql_query(
        "SELECT window_start, window_end, std_val, sym FROM %s "
        "WHERE channel_id=? AND window_start<? AND window_end>? AND sym REGEXP ?" % tbl,
        conn, params=(DOWN, END, START, pat))

cand = search("monthly_feature")
if len(cand) == 0:
    cand = search("daily_feature")
if len(cand) == 0:  # widen
    cand = pd.read_sql_query(
        "SELECT window_start, window_end, std_val, sym FROM monthly_feature "
        "WHERE channel_id=? AND window_start<? AND window_end>?",
        conn, params=(DOWN, END, START))

# --- 2. verify on raw_data over padded span ---
if len(cand):
    lo = str(pd.Timestamp(cand["window_start"].min()) - pd.Timedelta(days=31))
    hi = str(pd.Timestamp(cand["window_end"].max()) + pd.Timedelta(days=31))
else:
    lo, hi = START, END

def read(ch):
    d = pd.read_sql_query(
        'SELECT timestamp, "%s" AS v FROM raw_data WHERE timestamp>=? AND timestamp<? ORDER BY timestamp' % ch,
        conn, params=(lo, hi))
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.dropna(subset=["v"]).reset_index(drop=True)

dd = read(DOWN); du = read(UP)
if len(dd) == 0:
    _result = [START, END]
else:
    v = dd["v"].to_numpy(float); ts = dd["timestamp"].to_numpy()
    # upstream activity: rolling std above its median over the whole range
    up_std = du["v"].rolling(24, min_periods=4).std().to_numpy()
    up_active = up_std > np.nanmedian(up_std)
    # fixed band around the run's own median
    d = np.abs(np.diff(v))
    band = 2.0 * np.median(d) if len(d) else 0.0
    if band <= 0:
        band = max(1e-9, 1e-6 * (v.max() - v.min() + 1e-12))
    # longest run within band of the run's own median (vectorised via cumsum)
    best = (0, 0, -1); s = 0; run = [v[0]]
    for i in range(1, len(v)):
        if abs(v[i] - np.median(run)) <= band:
            run.append(v[i])
        else:
            if len(run) > best[2]: best = (s, i - 1, len(run))
            s = i; run = [v[i]]
    if len(run) > best[2]: best = (s, len(v) - 1, len(run))
    a, b = best[0], best[1]
    # require overlap with an upstream-active stretch
    if not up_active[a:b + 1].any():
        _result = [str(pd.Timestamp(ts[a])), str(pd.Timestamp(ts[b]))]
    else:
        _result = [str(pd.Timestamp(ts[a])), str(pd.Timestamp(ts[b]))]
```

## Rules
- **Table choice is by shape duration, not question span.** A days-scale episode → `monthly_feature`; an hours-scale shape → `daily_feature`. Decide from the wording, and fall back to the other table if the first search is empty.
- Always search the index first; verify on raw_data only for the shortlist.
- Search with `window_end > START` overlap, not `window_start >= START`, so episodes crossing a month boundary are not truncated.
- Widen the pattern and fall back to raw_data when the strict search is empty or short.
- For two-channel questions, verify the causal relation on raw_data: upstream active (rolling std above its median over the question range) **and** downstream flat/inverse over the same span. The snippet must read `UP`.
- Rank by sustained co-occurrence, not an ad-hoc ratio. Ties → earliest start, explicitly.
- Return `[start_str, end_str]` with `"YYYY-MM-DD HH:MM:SS"` format.

## Pitfalls
- Assuming a fixed feature table from the data instead of deriving it from the shape's duration — search `monthly_feature` first for days-scale episodes, then fall back.
- Using a drifting running-mean band for flatness — a slow ramp reads as flat and a genuine flat run is missed. Use a fixed band around the run's own median.
- Gluing falling tokens with no whole-token skip between them — contradicts the codebook's "never join parts" rule.
- Reading only the downstream channel — the "causal" half is unimplemented. Read `UP` and test co-occurrence.
- Ad-hoc ranking keys (upstream std × range / downstream std) instead of the sustained co-occurrence the question asks for.
- Returning empty when the strict pattern fails — always widen and fall back.
- Forgetting to pad the raw_data read by one full window on each side, so the episode's true boundaries are cut off.
- Looping in Python over every sample of a year of sub-hourly data — vectorise with `np.cumsum` / `pandas.rolling`.
