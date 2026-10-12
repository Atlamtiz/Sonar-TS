---
id: extreme-hydrological-episode-interval
description: Use when the question asks to identify the time interval within a year during which a channel experienced its most significant extreme hydrological episode of a named type (flood, drought, surge, dry-out, historically high/low level), where the episode is a sustained rise-then-plateau (or slow-drift-then-low-plateau) event and the answer is a start/end timestamp pair; not for shorter sub-monthly upward spikes, not for single-day or single-hour anomaly detection, not for segmenting a month into trend stages, and not for simple min/max/mean summaries.
---

# Extreme episode: locate by monthly shape, rank by sustained deviation area

## When to use
The question names a channel, a year, and an extreme episode type (flood, surge, high level, drought, dry-out) and asks for the interval of the most significant such episode. The episode is a sustained event: a rise to a plateau (flood/surge/high) or a slow drift down to a low plateau (drought/dry-out). It is NOT for single-day spikes, NOT for segmenting a whole month into labelled stages plus outliers, and NOT for plain min/max/mean summaries.

## Algorithm
1. Read from the question: `<channel>`, `<year>`, `<episode_type>`. Set `SIGN = +1` for flood/surge/high, `SIGN = -1` for drought/dry-out/dry. The answer is one interval.
2. **Search `monthly_feature`** — the shape spans many days, so 1 token = 1 day is the granularity that makes a plateau visible (a `daily_feature` window is one day, too short). Build a loose sym pattern:
   - flood/surge/high: `(?:[a-e]R[012][=~])+(?:[a-e][RFS][012][=~])*?(?:[de]S[012][=~]){2,}` — the plateau is a run of `S` tokens at high level letters; the codebook says a plateau is `S` at any level, so leave the rate slot open.
   - drought/dry-out: `(?:[a-e][RFS][012][=~])*?(?:[ab]S[012][=~]){3,}` — a drought is a long low plateau reached by a slow drift down; the drift reads as `S` tokens with descending level letters, so the pattern must NOT require F tokens and must NOT require a preceding high-level token. It only requires a run of at least three low stable tokens.
   Filter `channel_id = <channel>` and `window_start` inside the year. If the strict pattern returns fewer than 1 candidate, re-run with the plateau run shortened to `{2,}` and the level class widened to `[ab][RFS][012][=~]` (drought) / `[de][RFS][012][=~]` (flood). Never stop at an empty strict search.
3. **Rank candidates on the feature table BEFORE capping.** Sort by `min_val` ascending (drought) or `max_val` descending (flood/surge/high), then `head(MAX_CAND = 20)`. Capping before ranking keeps the calendar-first months and drops the true extreme month.
4. **Verify on raw_data** for each shortlisted monthly window. Guard the companion column: `cols = [r[1] for r in conn.execute("PRAGMA table_info(raw_data)")]`; if `<channel>_hist_mean` is absent, use a **centered** rolling baseline (`v.rolling(30, min_periods=1, center=True).mean()`) — a trailing mean biases the deviation toward the start of the year. Read `timestamp, "<channel>", "<channel>_hist_mean"` over `[window_start - 45d, window_end + 45d]` so an episode straddling a month border is not truncated. Compute `d = SIGN * (v - h)`, `dropna()`.
5. **Ranking key = sustained deviation area**, not the single deepest deviation. For each candidate compute `peak = d.max()`; skip if `not np.isfinite(peak) or peak <= 0`. Then `area = sum(d[d >= 0.5*peak])` — depth × duration of the sustained episode. A short sharp excursion has a large peak but a small area; a long flat low plateau has a smaller peak but a large area, which is what "most significant drought" means. Ties broken by longer plateau duration (count of samples above the half-peak threshold).
6. **Expand to the full episode**: from the peak timestamp, walk outward while `d` stays above `0.5 * peak`, measured from zero, not from `min(d)`. Restrict to the **contiguous run containing the peak** — do not take the first and last above-threshold sample in the slice, which would span a gap between two separate excursions. If the interval is shorter than the plateau the question implies, lower to `0.25 * peak` and re-expand.
7. Return `[start_timestamp, end_timestamp]` as `"YYYY-MM-DD HH:MM:SS"` strings (the eval_metric is `iou` and expects full timestamps).

```python
import pandas as pd, numpy as np

CH = "<channel>"
YEAR = <year>
KIND = "<episode_type>"          # flood | surge | high | drought | dry-out | dry
SIGN = -1 if KIND in ("drought", "dry-out", "dry") else 1
HIST = CH + "_hist_mean"
MAX_CAND = 20

cols = [r[1] for r in conn.execute("PRAGMA table_info(raw_data)")]
has_hist = HIST in cols

if SIGN > 0:
    pat = r"(?:[a-e]R[012][=~])+(?:[a-e][RFS][012][=~])*?(?:[de]S[012][=~]){2,}"
else:
    pat = r"(?:[a-e][RFS][012][=~])*?(?:[ab]S[012][=~]){3,}"

q = """SELECT window_start, window_end, sym, min_val, max_val FROM monthly_feature
       WHERE channel_id = ? AND window_start >= ? AND window_start < ?
         AND sym REGEXP ?"""
cand = pd.read_sql_query(q, conn, params=[CH, f"{YEAR}-01-01", f"{YEAR+1}-01-01", pat])
if len(cand) < 1:
    loose = pat.replace("{3,}", "{2,}").replace("[ab]S[012]", "[ab][RFS][012]") \
               .replace("[de]S[012]", "[de][RFS][012]")
    cand = pd.read_sql_query(q, conn, params=[CH, f"{YEAR}-01-01", f"{YEAR+1}-01-01", loose])
if len(cand) < 1:
    cand = pd.read_sql_query(
        """SELECT window_start, window_end, sym, min_val, max_val FROM monthly_feature
           WHERE channel_id = ? AND window_start >= ? AND window_start < ?""",
        conn, params=[CH, f"{YEAR}-01-01", f"{YEAR+1}-01-01"])
cand = cand.sort_values("min_val" if SIGN < 0 else "max_val",
                        ascending=(SIGN < 0)).head(MAX_CAND)

def load(ws, we):
    lo = (pd.Timestamp(ws) - pd.Timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")
    hi = (pd.Timestamp(we) + pd.Timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")
    sel = f'SELECT timestamp, "{CH}" AS v' + (f', "{HIST}" AS h' if has_hist else '') + \
          ' FROM raw_data WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp'
    df = pd.read_sql_query(sel, conn, params=[lo, hi])
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    v = df["v"].astype(float)
    if has_hist:
        d = SIGN * (v - df["h"].astype(float))
    else:
        d = SIGN * (v - v.rolling(30, min_periods=1, center=True).mean())
    d = d.dropna()
    return (df, d) if not d.empty else None

best = None
for _, row in cand.iterrows():
    got = load(row["window_start"], row["window_end"])
    if got is None: continue
    df, d = got
    peak = float(d.max())
    if not np.isfinite(peak) or peak <= 0: continue
    area = float(d[d >= 0.5 * peak].sum())
    if best is None or area > best["area"]:
        best = {"area": area, "peak": peak, "df": df, "d": d}

if best is None:
    got = load(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
    if got is None:
        _result = [f"{YEAR}-01-01 00:00:00", f"{YEAR}-12-31 23:59:59"]
    else:
        df, d = got
        peak = float(d.max())
        if not np.isfinite(peak) or peak <= 0:
            _result = [f"{YEAR}-01-01 00:00:00", f"{YEAR}-12-31 23:59:59"]
        else:
            best = {"area": float(d[d >= 0.5 * peak].sum()), "peak": peak, "df": df, "d": d}

if best is not None:
    d = best["d"]; ts = best["df"]["timestamp"].loc[d.index]; peak = best["peak"]
    pos = int(np.argmax(d.values))
    lo_i = hi_i = pos
    for frac in (0.5, 0.25):
        thr = frac * peak
        mask = (d.values >= thr)
        lo_i = pos
        while lo_i > 0 and mask[lo_i - 1]: lo_i -= 1
        hi_i = pos
        while hi_i < len(mask) - 1 and mask[hi_i + 1]: hi_i += 1
        if hi_i > lo_i: break
    _result = [str(ts.iloc[lo_i]), str(ts.iloc[hi_i])]
```

## Rules
- The episode is a sustained plateau shape → search `monthly_feature` (1 token = 1 day). Never shortlist from `daily_feature` for this question type.
- **The drought pattern must not require F tokens and must not require a preceding high-level token.** A drought is a long low plateau reached by a slow drift; the drift reads as `S` tokens with descending level letters. The pattern is just a run of low stable tokens.
- **Rank candidates on the feature table before capping at `MAX_CAND`** — sort by `min_val` ascending (drought) or `max_val` descending (flood/surge/high). Capping first keeps the calendar-first months.
- The ranking key is the **sustained deviation area** — the sum of the deviation over the interval where it stays above half its peak — not the single deepest deviation, not `max_val`, not `max_val - min_val`. A short sharp excursion must not beat a long flat plateau.
- "Historically high/low" means deviation from `<channel>_hist_mean`; read the companion column when it exists, guard with `PRAGMA table_info`, fall back to a **centered** rolling baseline otherwise.
- The returned interval must span the whole episode (drift + plateau), found by expanding from the peak while the deviation stays above half the peak, restricted to the **contiguous run containing the peak**; if too short, lower to a quarter.
- Guard `peak` with `np.isfinite(peak) and peak > 0` before ranking or expanding — NaN or zero peaks otherwise propagate into the final mask.
- Return full timestamps `"YYYY-MM-DD HH:MM:SS"` (the eval_metric is `iou`), not date-only strings.
- Keep the sym pattern loose; widen slots before falling back to a raw scan. Cap verification at `MAX_CAND`.

## Pitfalls
- Requiring F tokens in the drought pattern: a slow drift down reads as `S` tokens with descending level letters, so an F-requiring pattern matches nothing and the fallback picks the wrong month.
- Requiring a preceding high-level token in the drought pattern: a drought can start from a mid-level baseline, so the pattern loses true windows.
- Capping `head(MAX_CAND)` before ranking: the loose pattern matches nearly every month, so the shortlist is the calendar-first 20 and the true drought month is dropped.
- Ranking by peak deviation alone: a short sharp excursion in a volatile month beats a long flat low plateau. Use the sustained area.
- Ranking by `min_val` / `avg_val` in the fallback: selects the month with the lowest absolute level, not the deepest sustained deficit below the historical mean.
- Threshold measured from `min(d)` instead of zero: a window with a high baseline expands to almost the whole window. Use `0.5 * peak` from zero.
- Taking the first and last above-threshold sample in the slice: if the slice contains two separate excursions the returned interval spans the gap. Restrict to the contiguous run containing the peak.
- Truncating the interval to the shortlisted monthly window: an episode straddling a month border is cut off. Read ±45 days around the window.
- Trailing rolling baseline: biases the deviation toward the start of the year. Use `center=True`.
- Assuming `_hist_mean` exists: guard with `PRAGMA table_info` or the SQL raises and the script scores zero.
- NaN in `v`/`h` propagating into `d.max()`: `dropna()` before ranking, and guard `np.isfinite(peak)`.
- Inclusive/exclusive ends: `window_start` inclusive, `window_end` exclusive; the returned interval is inclusive of both endpoints.
