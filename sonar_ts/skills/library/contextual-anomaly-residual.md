---
id: contextual-anomaly-residual
description: For "identify the period that experienced the most significant {flood/surge/drought/dry-out/etc.}" — load the WHOLE YEAR of raw_data and its "<ch>_hist_mean" companion, compute residual, then find the LONGEST CONTIGUOUS RUN of residual past a 2σ threshold (events typically span 60-90 days, often crossing month boundaries).
---

# Contextual Anomaly — full-year residual scan

## The hidden companion column

In this subtask, `raw_data` carries TWO numeric columns per task: the channel under analysis (e.g. `"67"`) AND its seasonal baseline `"67_hist_mean"`. The anomaly signal is

residual(t) = raw[t] - hist_mean[t]

A flood / surge / "historically high level" → residual > +K·σ_residual. A drought / dip / "dry-out" / "abnormally low" → residual < −K·σ_residual.

## CRITICAL: scan the ENTIRE YEAR, not a single month

Ground-truth events span **60-90 days** and frequently cross month boundaries (e.g. Aug 14 → Nov 12). DO NOT:

* query `monthly_feature` first to pick a "best month" and then bound the search to that one month — the event almost always overflows it,
* use `daily_feature` to score days independently then take the top — the answer is a SINGLE contiguous interval, not a date set.

The reliable algorithm is one full-year raw_data scan + a contiguous-run detector. The scan is ~35k rows for 15-min data — sub-second in pandas.

## Algorithm

```python
import re

YEAR = <year>            # parsed from question
CH   = "<ch>"            # the channel mentioned in the question
HM   = f"{CH}_hist_mean"

# 1. Pull the WHOLE year, both columns, into one DataFrame.
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}", "{HM}" FROM raw_data '
    "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
    conn, params=(f"{YEAR}-01-01", f"{YEAR + 1}-01-01"))
df["timestamp"] = pd.to_datetime(df["timestamp"])

raw  = df[CH].astype(float).values
base = df[HM].astype(float).values
residual = raw - base

# 2. Direction from question phrasing.
question_low = question.lower()
positive_words = ("flood", "surge", "spike", "elevated", "above-normal",
                  "historically high", "high water", "wet")
negative_words = ("drought", "dry-out", "dry", "dip", "depression",
                  "below-normal", "historically low", "low water")
pos_score = sum(w in question_low for w in positive_words)
neg_score = sum(w in question_low for w in negative_words)
direction = +1 if pos_score >= neg_score else -1

# 3. Smooth the residual with a 1-day rolling mean so single-spike noise
# doesn't fragment the run.
step = df["timestamp"].diff().dropna().median()
per_day = max(1, int(round(pd.Timedelta(days=1) / step)))
smoothed = pd.Series(residual).rolling(per_day, center=True).mean().bfill().ffill().values

sigma = float(np.nanstd(smoothed))
if sigma < 1e-9:
    sigma = float(np.nanstd(raw)) * 0.1 or 1.0

# 4. Longest contiguous run above threshold. Start at 2σ and walk down
# until we get a run of at least ~30 days (the events are long).
def longest_run(mask: np.ndarray) -> tuple[int, int] | None:
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

best = None
min_run_length = 30 * per_day      # demand at least 30-day-long run
for k in (2.0, 1.5, 1.0, 0.75, 0.5, 0.3):
    threshold = k * sigma
    mask = (smoothed > threshold) if direction == +1 else (smoothed < -threshold)
    cand = longest_run(mask)
    if cand is not None and (cand[1] - cand[0]) >= min_run_length:
        best = cand
        break

if best is None:
    # Last resort: take a 60-day window centred on the residual's extreme.
    centre = int(np.argmax(np.abs(smoothed)))
    half = int(round(pd.Timedelta(days=30) / step))
    best = (max(0, centre - half), min(len(smoothed) - 1, centre + half))

a, b = best
_result = [df["timestamp"].iloc[a].strftime("%Y-%m-%d %H:%M:%S"),
           df["timestamp"].iloc[b].strftime("%Y-%m-%d %H:%M:%S")]
```

## Drought edge case

Some "dry-out" / "abnormally calm" questions describe events where the value collapses to near-zero variance instead of dropping below baseline. If the standard negative-direction scan fails (no run of 30+ days), try a secondary metric: longest run where the rolling std of `raw` itself is unusually small:

rstd = pd.Series(raw).rolling(per_day, center=True).std().bfill().ffill().values low_var_mask = rstd < 0.3 * np.nanmean(rstd) secondary = longest_run(low_var_mask)

Prefer the negative-direction result if it exists; only fall back to the variance signal when needed.

## Rules

* **One scan, whole year.** Don't pre-filter by month / by daily_feature.
* Always read BOTH columns (`<ch>` and `<ch>_hist_mean`).
* Smooth with a 1-day rolling mean before thresholding.
* The output interval should be ≥ 30 days wide — anything narrower means the threshold-walk fell through; re-check the direction.
* Output `[start, end]` as `%Y-%m-%d %H:%M:%S` strings.
