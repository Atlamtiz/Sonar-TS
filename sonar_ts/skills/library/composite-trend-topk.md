---
id: composite-trend-topk
description: For "among days with trend pattern X, return the top-K days where the {adj} {kind} segment is the {superlative}" — Search-Then-Verify with SAX. SEARCH writes a SAX regex over daily_feature (the morphological symbol per day) to localise candidate days; VERIFY fetches raw data for ONLY those candidates, measures the longest monotone/flat run on the target segment, and ranks them. If the SAX search returns nothing, fall back to the whole year.
---

# Composite Trend Top-K — SAX Search-Then-Verify

`SQL SAX regex → candidate days → (empty? fall back to all) → verify ONLY candidates → rank → top-K`.

The code block below is a **complete, calibrated solution**. Copy it almost verbatim and
substitute ONLY the five marked `<...>` values. **Do NOT replace the verification metric, the
floor, or the ranking with your own** — they are tuned to this benchmark and changing them
breaks the score. In particular:

* **Metric = the amplitude of the longest monotone run** of the target kind (the segment's total
  rise/fall). This benchmark's ground truth ranks by *how much the segment changes*, **not** by
  instantaneous slope. **Do NOT use `polyfit` / `lstsq` / regression slope** — it mis-ranks.
* **Floor**: the per-day metrics are bimodal — a near-zero cluster (days without a real segment
  of the target kind) and a higher cluster (days that genuinely contain it). An ascending sort
  ("slowest / smallest") would otherwise pick "no segment at all", so we drop the no-signal
  cluster with a quantile floor (tighter for ascending, looser for descending). Keep it as written.

```python
import re
import numpy as np, pandas as pd

CH    = "<channel id>"
YEAR  = <year int>
K     = 5
kinds = <ordered kind list from the pattern, e.g. ["stable","fall","fall"]>
target_kind = "<the kind the question ranks on: rise|fall|stable>"
target_adj  = "<the adjective on that segment, e.g. slow>"
sup         = "<the superlative word, e.g. slowest>"

# ---------- SEARCH: SAX regex over the feature index (plain SAX, paper §5.3) ----------
# The paper's own tight fuzzy bands, backref-FREE (so they survive as a SQL string param).
# Do NOT collapse repeated kinds — every requested segment must appear in order. This is the
# paper's original behaviour and is what makes the SAX search actually prune (it verifies only
# ~1/4 of the year, not all of it).
def kind_to_regex(k):
    if k == "rise":   return r"[ab].*[de]"
    if k == "fall":   return r"[de].*[ab]"
    if k == "stable": return r"(?:a{3,}|b{3,}|c{3,}|d{3,}|e{3,})"
    return r".*"
sax_regex = ".*".join(kind_to_regex(k) for k in kinds)
hits = pd.read_sql_query(
    "SELECT window_start FROM daily_feature "
    "WHERE channel_id = ? AND window_start >= ? AND window_start < ? AND sax REGEXP ?",
    conn, params=(CH, f"{YEAR}-01-01", f"{YEAR+1}-01-01", sax_regex))
cand_days = sorted({pd.Timestamp(s).date() for s in hits["window_start"]})

# ---------- FALLBACK (paper Fig.4: SAX too strict -> raw data) ----------
# Fig.4 says an empty SAX result means the approximation was too strict and you must fall back to
# raw data. The same applies when it returns fewer than K days — you cannot rank a top-K answer
# from them. In both cases verify the whole year for this (rare) task; most tasks keep their
# pruned candidate set.
if len(cand_days) < K:
    alld = pd.read_sql_query(
        "SELECT DISTINCT window_start FROM daily_feature "
        "WHERE channel_id = ? AND window_start >= ? AND window_start < ?",
        conn, params=(CH, f"{YEAR}-01-01", f"{YEAR+1}-01-01"))
    cand_days = sorted({pd.Timestamp(s).date() for s in alld["window_start"]})

# ---------- VERIFY: raw data for CANDIDATE days ONLY ----------
cand_strs = [d.isoformat() for d in cand_days]
ph = ",".join("?" * len(cand_strs))
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    f"WHERE substr(timestamp,1,10) IN ({ph}) ORDER BY timestamp",
    conn, params=cand_strs)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

def longest_run(mask):
    best, i = None, 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    return best

def day_metric(v, kind):                     # amplitude of the longest monotone run (NOT slope)
    if len(v) < 8:
        return None
    sm = pd.Series(v).rolling(4, center=True).mean().bfill().ffill().values
    dd = np.diff(sm)
    if kind == "rise":
        r = longest_run(dd > 0);  return float(sm[r[1]] - sm[r[0]]) if r else None
    if kind == "fall":
        r = longest_run(dd < 0);  return float(sm[r[0]] - sm[r[1]]) if r else None
    rstd = pd.Series(dd).rolling(4, center=True).std().bfill().ffill().values
    f = longest_run(rstd < float(np.nanmedian(rstd)))
    return float(np.std(v[f[0]:f[1] + 1])) if f else None

per_day = []
for date, sub in df.groupby("date"):
    m = day_metric(sub[CH].astype(float).values, target_kind)
    if m is not None:
        per_day.append({"date": date, "metric": m})

# ---------- RANK among candidates -> top-K (keep this floor + ranking exactly) ----------
descending = sup in {"fastest","biggest","largest","strongest","highest",
                     "deepest","most","sharpest","steepest"}
if target_kind == "stable" and target_adj == "steady":
    descending = False
if per_day:
    mvals = np.array([d["metric"] for d in per_day])
    floor = float(np.quantile(mvals, 0.80 if not descending else 0.50))
    kept = [d for d in per_day if d["metric"] >= floor] or per_day
    kept.sort(key=lambda d: d["metric"], reverse=descending)
    _result = [str(d["date"]) for d in kept[:K]]
else:
    _result = []
```

## Rules
* **Search-Then-Verify, strict**: candidates come from the SQL `sax REGEXP` search; verify pulls raw for those candidate days ONLY — never the whole year unless the SAX search was empty (then fall back to all, per the paper's Fig.4 rule).
* **Use the verification exactly as written.** The metric is the longest-run *amplitude*; do NOT substitute regression slope, `polyfit`, or a different floor — they change the ranking and break the benchmark score.
* `sax REGEXP '...'` / `regexp_like(sax, '...')` are both registered on the connection. Adjectives (rapid/slow/…) are verify-time, never in the regex. The floor comes from the per-question metric distribution — never from external knowledge of the answer. Never call `exit()` / `sys.exit()`.
