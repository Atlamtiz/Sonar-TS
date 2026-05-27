---
id: causal-anomaly-correlation
description: For "where channel X_down shows a causal anomaly against the upstream X_up" — branch on the anomaly type stated in the question: "inverse trend" uses regression-residual; "flat line during high activity" uses (downstream_rolling_std small AND upstream_rolling_std large).
---
# Causal Anomaly — two-mode detection

## The setup

L3-T3 tasks always have two channels in `raw_data`: `<X>_up_<year>` and `<X>_down_<year>`. The question states the anomaly type explicitly:

* **"inverse trend against the source"** — downstream slope flips; the linear relationship `down ≈ a + b·up` breaks via sign reversal.
* **"flat line during high activity"** — downstream becomes nearly constant while upstream still varies; the relationship breaks via a collapse of downstream variance.

These two modes need DIFFERENT detectors — a single metric (rolling correlation, regression residual) handles one cleanly and the other poorly.

## Algorithm — branch by question text

```python
import re

YEAR = <year>
UP   = "<X>_up_<year>"
DOWN = "<X>_down_<year>"
question_low = question.lower()

# 1. Pull both channels for the full year.
df = pd.read_sql_query(
    f'SELECT timestamp, "{UP}", "{DOWN}" FROM raw_data '
    "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
    conn, params=(f"{YEAR}-01-01", f"{YEAR + 1}-01-01"))
df["timestamp"] = pd.to_datetime(df["timestamp"])
u = df[UP].astype(float).values
d = df[DOWN].astype(float).values

step = df["timestamp"].diff().dropna().median()

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

# 2. Detect anomaly type and apply the matched detector.
if "flat" in question_low or "constant" in question_low or "still" in question_low:
    # === FLAT-LINE MODE ============================================
    # Signature: rolling-std of downstream is SMALL while rolling-std
    # of upstream is LARGE. Use a ~6-hour window.
    W = max(4, int(round(pd.Timedelta(hours=6) / step)))
    rstd_d = pd.Series(d).rolling(W, center=True).std().bfill().ffill().values
    rstd_u = pd.Series(u).rolling(W, center=True).std().bfill().ffill().values

    # Reference scales (whole-year medians).
    med_d = float(np.nanmedian(rstd_d)) if np.any(rstd_d > 0) else 1e-9
    p75_u = float(np.nanpercentile(rstd_u, 75))

    # The anomaly window has down-std much smaller than its yearly median
    # AND up-std at least at the year's 75th percentile.
    min_run = max(W, int(round(pd.Timedelta(days=2) / step)))
    best = None
    for q_low, q_high in [(0.20, 0.50), (0.30, 0.40), (0.40, 0.30), (0.50, 0.20)]:
        mask = (rstd_d < q_low * med_d) & (rstd_u > p75_u * q_high * 2)
        cand = longest_run(mask)
        if cand is not None and (cand[1] - cand[0]) >= min_run:
            best = cand
            break

    # Backup: just (down_std < small) without the upstream activity check.
    if best is None:
        for q_low in (0.20, 0.30, 0.50, 0.80):
            mask = rstd_d < q_low * med_d
            cand = longest_run(mask)
            if cand is not None and (cand[1] - cand[0]) >= min_run:
                best = cand
                break
else:
    # === INVERSE-TREND MODE ========================================
    # Closed-form OLS over the full year, then large |residual| run.
    um = np.nanmean(u); dm = np.nanmean(d)
    denom = float(np.nansum((u - um) ** 2))
    b = float(np.nansum((u - um) * (d - dm)) / denom) if denom > 1e-9 else 0.0
    a = dm - b * um
    residual = d - (a + b * u)

    W = max(2, int(round(pd.Timedelta(hours=6) / step)))
    abs_res = np.abs(pd.Series(residual).rolling(W, center=True).mean()
                      .bfill().ffill().values)
    sigma = float(np.nanstd(residual))
    if sigma < 1e-9:
        sigma = float(np.nanstd(d)) * 0.05 or 1.0

    min_run = max(W, int(round(pd.Timedelta(days=2) / step)))
    best = None
    for k in (3.0, 2.5, 2.0, 1.5, 1.0):
        mask = abs_res > k * sigma
        cand = longest_run(mask)
        if cand is not None and (cand[1] - cand[0]) >= min_run:
            best = cand
            break

# 3. Last-resort fallback.
if best is None:
    centre = len(d) // 2
    half = int(round(pd.Timedelta(days=3) / step))
    best = (max(0, centre - half), min(len(d) - 1, centre + half))

a_idx, b_idx = best
_result = [df["timestamp"].iloc[a_idx].strftime("%Y-%m-%d %H:%M:%S"),
           df["timestamp"].iloc[b_idx].strftime("%Y-%m-%d %H:%M:%S")]
```

## Why two detectors

Regression residual measures *how far downstream is from its linear prediction*. For inverse-trend anomalies this spikes (residual is twice the upstream's variation magnitude during the inverted segment). For flat-line anomalies, downstream is constant — the residual then equals `const − b·u(t)`, which tracks the upstream's natural variation and does NOT spike in any localised way. So flat-line needs a different signal: downstream variance vs upstream variance.

## Rules

* Output interval should be 3-10 days (the planted anomalies are short).
* Always parse the anomaly TYPE first (flat / inverse) and branch.
* Use raw_data only.
