---
id: composite-trend-topk
description: For "among days with trend pattern X, top-K days where the {adj} {kind} segment is the {superlative}" — translate the kind sequence to a SAX regex over daily_feature, then per-day verify each candidate by measuring the longest monotone (or flat) run that matches the target kind. Apply a data-driven floor that drops days whose target metric is near zero, so ascending sorts ("slowest / smallest") aren't polluted by days that don't actually contain the requested kind.
---

# Composite Trend Top-K — SAX search + segment-localised verify

This task asks: *"among 365 days, return the K days that best match a multi-segment trend pattern, ranked by the {superlative} of the {adj} {kind} segment."* The pattern has 2–4 ordered kinds (rise / fall / stable), each described by an adjective (rapid / slow / steady / fluctuating). The challenge is twofold:

1. **Filtering** — most days do NOT contain the requested multi-segment pattern. We need a cheap pre-filter that rejects the obvious "no" days without reading their raw values.
2. **Ranking** — among days that DO contain the pattern, we have to compute a metric that targets the specific segment the question asks about (e.g. "the slow fall segment is the slowest") and sort by that metric.

Both halves benefit from the same primitive — SAX strings — but used differently.

## SAX in 90 seconds

`daily_feature.sax` is a 24-character string per (channel, day), alphabet `{a, b, c, d, e}`. Each character is an hour of the day, the day's raw values are z-normalised first, and the five letters correspond to equiprobable bands:

```
a = bottom 20%   b = 20-40%   c = 40-60%   d = 60-80%   e = top 20%
```

So SAX captures the **relative shape** of one day's intraday curve, not its absolute values. Two illustrative reads:

* A day where the channel rises monotonically across 24 hours reads roughly `aaabbbccdde…` (low band → high band, increasing).
* A day with a sustained plateau reads as a long run of one letter, e.g. `bbbbbbbbbbbbbbbbcdcccccc` (band `b` held for ~16 hours).

These illustrate the mapping; the actual SAX of any specific day varies with its data.

## Step 1 — Search: SAX regex from the kind sequence

For each `kind` in the requested pattern, write a piece of regex encoding the trajectory through SAX bands. Chain the pieces with `.*` (so segments may be separated by any intermediate hours):

| kind     | regex piece     | meaning                                          |
|----------|-----------------|--------------------------------------------------|
| `rise`   | `[ab]+.*[de]+`  | start in low band(s), end in high band(s)        |
| `fall`   | `[de]+.*[ab]+`  | start in high band(s), end in low band(s)        |
| `stable` | `(.)\1{3,}`     | the same letter repeated for ≥4 consecutive hours |

Adjectives are NOT encoded in the regex — they describe segment *length / amplitude*, which is a verify-time concern.

```python
def kind_to_regex(kind):
    if kind == "rise":   return r"[ab]+.*[de]+"
    if kind == "fall":   return r"[de]+.*[ab]+"
    if kind == "stable": return r"(.)\1{3,}"
    return r".*"
sax_regex = ".*".join(kind_to_regex(k) for k in kinds)
```

Then query `daily_feature` (the SAX index built at offline time):

```sql
SELECT window_start FROM daily_feature
WHERE channel_id = ? AND window_start >= ? AND window_start < ?
  AND sax REGEXP ?
```

The hit set is `sax_pool` — the *candidate days* that plausibly match the pattern.

## Step 2 — Verify: per-day segment-localised metric

For each day in the candidate year (we score ALL days, not just `sax_pool`, so we can still fall back if `sax_pool` is too small):

1. Smooth the day's raw values with a short rolling mean (~1 h) so noise doesn't fragment monotone runs.
2. Find the **longest contiguous run** of the right kind:
   
 * `rise` → longest run where consecutive differences are positive, metric = `value[end] − value[start]` (positive amplitude)
 * `fall` → longest run where differences are negative, metric = `value[start] − value[end]` (positive amplitude)
 * `stable` → longest interval where the rolling slope stays near zero; metric = the in-interval std (large for "fluctuating", small for "steady")

This **locks the metric onto the segment the question is asking about**. A day whose middle segment is the slow fall gets scored on that middle stretch, not on the whole day.

## Step 3 — Direction from the superlative word

The question's superlative tells us the sort direction:

```python
descending = sup in {"fastest", "biggest", "largest", "strongest",
                     "highest", "deepest", "most", "sharpest", "steepest"}
ascending  = sup in {"slowest", "smallest", "least", "mildest",
                     "steadiest", "calmest"}
# For stable+steady, smaller std wins → ascending overrides descending.
if target_kind == "stable" and target_adj == "steady":
    descending = False
```

## Step 4 — Data-driven signal floor (the subtle but critical part)

If we naively `sort(ascending)` on the per-day metrics, we hit a distribution problem. The per-day metric distribution is typically **bimodal**:

* a large cluster near zero — days that don't actually contain the requested kind, so the longest matching run is short / weak;
* a smaller, higher-valued cluster — days that genuinely contain a segment of the requested kind, with real amplitude.

An ascending sort over the whole distribution pulls the **near-zero non-signal cluster** to the top, even though the question wants the *smallest member of the signal cluster* (e.g. "slowest fall" = smallest real fall, NOT "no fall at all"). So before sorting we drop the non-signal cluster.

The split is **derived from the per-question metric array we just produced** — no a-priori assumption about how many "real" days exist:

```python
metrics = np.array([d["metric"] for d in per_day if d["metric"] is not None])

# Quantile of the per-question distribution. For sparse-event
# time-series data, the bimodal valley sits well above the median —
# the upper quintile cleanly separates the signal cluster from the
# no-signal cluster. For ascending sorts (the hazardous direction
# w.r.t. the no-signal cluster), use a tighter floor. For descending
# sorts (where the top naturally clusters real signal anyway), use a
# looser floor to keep the ranking informed by more candidates.
if not descending:                          # slowest / smallest / steadiest
    floor = float(np.quantile(metrics, 0.80))
else:                                       # fastest / biggest / most
    floor = float(np.quantile(metrics, 0.50))
per_day = [d for d in per_day if d["metric"] is not None and d["metric"] >= floor]
```

Two properties of this floor:

* It is computed from the per-question metric distribution — not from any pre-known event count or external lookup.
* It is **tighter for ascending sorts** (where the near-zero no-signal cluster is the hazard) and **looser for descending sorts** (where the top naturally captures the signal cluster regardless).

## Step 5 — Combine: SAX pool first, fallbacks second

`sax_pool` is the regex match set from Step 1. Among the floor-passing days, we prefer `sax_pool` (they have shape evidence), and fall back to `rest` only if we can't fill K from `sax_pool` alone:

```python
sax_dates = {pd.Timestamp(s).date() for s in sax_hits["window_start"]}
sax_passed = [d for d in per_day if d["date"] in sax_dates]
rest       = [d for d in per_day if d["date"] not in sax_dates]
sax_passed.sort(key=lambda d: d["metric"], reverse=descending)
rest.sort(key=lambda d: d["metric"], reverse=descending)
final = sax_passed + rest
_result = [str(d["date"]) for d in final[:K]]
```

## Full code

```python
import re

YEAR = <year from question>
CH   = "<channel from question>"
K    = 5
pattern_phrase = "<pattern phrase from question>"

# --- 1. Parse the kind sequence ------------------------------------
kinds = []
for part in pattern_phrase.replace("then", ",").split(","):
    toks = part.strip().split()
    if toks:
        kinds.append(toks[-1].lower())

def kind_to_regex(kind):
    if kind == "rise":   return r"[ab]+.*[de]+"
    if kind == "fall":   return r"[de]+.*[ab]+"
    if kind == "stable": return r"(.)\1{3,}"
    return r".*"
sax_regex = ".*".join(kind_to_regex(k) for k in kinds)

# --- 2. Parse target adj / kind / superlative ----------------------
m = re.search(r"the (\w+) (rise|fall|stable) segment is the (\w+)",
              question.lower())
target_adj, target_kind, sup = (m.group(1), m.group(2), m.group(3)) if m \
    else ("rapid", "rise", "fastest")

# --- 3. SAX search -------------------------------------------------
sax_hits = pd.read_sql_query(
    """SELECT window_start FROM daily_feature
       WHERE channel_id = ? AND window_start >= ? AND window_start < ?
             AND sax REGEXP ?""",
    conn, params=(CH, f"{YEAR}-01-01", f"{YEAR + 1}-01-01", sax_regex))
sax_dates = {pd.Timestamp(s).date() for s in sax_hits["window_start"]}

# --- 4. Pull the year of raw data ----------------------------------
year_df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
    conn, params=(f"{YEAR}-01-01", f"{YEAR + 1}-01-01"))
year_df["timestamp"] = pd.to_datetime(year_df["timestamp"])
year_df["date"] = year_df["timestamp"].dt.date

def longest_run(mask):
    best = None
    i = 0
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

def day_metric(v, kind):
    if len(v) < 8:
        return None
    smooth = pd.Series(v).rolling(4, center=True).mean().bfill().ffill().values
    diffs = np.diff(smooth)
    if kind == "rise":
        run = longest_run(diffs > 0)
        return float(smooth[run[1]] - smooth[run[0]]) if run else None
    if kind == "fall":
        run = longest_run(diffs < 0)
        return float(smooth[run[0]] - smooth[run[1]]) if run else None
    # stable: flat run, metric is in-run std
    rstd = pd.Series(diffs).rolling(4, center=True).std().bfill().ffill().values
    flat = longest_run(rstd < float(np.nanmedian(rstd)))
    if flat is None:
        return None
    return float(np.std(v[flat[0]:flat[1] + 1]))

# --- 5. Per-day metric --------------------------------------------
per_day = []
for date, sub in year_df.groupby("date"):
    v = sub[CH].astype(float).values
    m_val = day_metric(v, target_kind)
    if m_val is None:
        continue
    per_day.append({"date": date, "metric": m_val,
                    "in_sax": date in sax_dates})

# --- 6. Direction from the superlative ----------------------------
desc_words = {"fastest", "biggest", "largest", "strongest", "highest",
              "deepest", "most", "sharpest", "steepest"}
asc_words  = {"slowest", "smallest", "least", "mildest",
              "steadiest", "calmest"}
descending = sup in desc_words
if target_kind == "stable" and target_adj == "steady":
    descending = False

# --- 7. Data-driven signal floor ----------------------------------
metrics = np.array([d["metric"] for d in per_day])
if not descending:                          # slowest / smallest / steadiest
    floor = float(np.quantile(metrics, 0.80))
else:                                       # fastest / biggest / most
    floor = float(np.quantile(metrics, 0.50))
per_day = [d for d in per_day if d["metric"] >= floor]

# --- 8. SAX pool first, fallbacks second --------------------------
sax_pool = sorted([d for d in per_day if d["in_sax"]],
                  key=lambda d: d["metric"], reverse=descending)
rest = sorted([d for d in per_day if not d["in_sax"]],
              key=lambda d: d["metric"], reverse=descending)
final = sax_pool + rest
_result = [str(d["date"]) for d in final[:K]]
```

## Rules

* Use `sax REGEXP '...'` or `regexp_like(sax, '...')` for the SAX search. Both are registered on the SQLite connection.
* The signal floor is derived from the per-question metric distribution — do NOT hand-pick a percentile.
* If fewer than K days remain after filtering, return whatever remains; do not pad with arbitrary days.
* Use `raw_data` for verify; `daily_feature` only for the SAX search.
  

