---
id: insight-synthesis
description: For "analyze the behavior of channel X for <month>" — segment the month with adaptive changepoint count (BIC over 1/2/3 cuts), refine each boundary by sliding ±N steps to minimise piecewise-linear SSE, classify segment kind by slope direction, and classify ADJ by within-month relative ranking (steepest rise/fall is "rapid"; most-fluctuating stable is "fluctuating"). One significant outlier via z-score residual.
---

# Insight Synthesis — adaptive segmentation, refined boundaries, within-month adj

## Output schema (CRITICAL)

`_result` must be a Python dict, not a string:

```python
_result = {
    "trend_segments": [
        {"start": "YYYY-MM-DD HH:MM:SS",
         "end":   "YYYY-MM-DD HH:MM:SS",
         "adj":   "<one of: gradual, rapid, steady, fluctuating>",
         "kind":  "<one of: rise, fall, stable>"},
        # 2-4 entries in chronological order
    ],
    "outliers": [
        {"timestamp": "YYYY-MM-DD HH:MM:SS", "value": <float>}
        # 0-1 entries (engine plants exactly one significant spike)
    ]
}
```

Post-processing renders this dict into the natural-language report. **Do NOT format the text yourself.**

## Three ideas that compound

1. **Adaptive n_cuts** — the GT has 2-4 stages per month. Try 1/2/3 cuts (giving 2/3/4 segments) and pick the one whose piecewise-linear SSE plus a BIC penalty is smallest.
2. **Boundary refinement** — the second-derivative argmax gives an approximate cut location; sliding the cut by ±a-few steps and accepting the position that reduces total SSE tightens the IoU of matched segments substantially.
3. **Within-month adj ranking** — "rapid" / "gradual" / "fluctuating" / "steady" are RELATIVE descriptors. The engine picks adj relative to a month's stages; an absolute slope threshold misclassifies channels with different scales. Within a month, "rapid" = the steepest rise/fall; "fluctuating" = the highest-std-after-detrend stable.

## Algorithm

```python
import re

m = re.search(r"(\d{4})-(\d{2})", question)
YEAR, MONTH = int(m.group(1)), int(m.group(2))
CH = "<channel from question>"

start = pd.Timestamp(f"{YEAR}-{MONTH:02d}-01")
end   = start + pd.offsets.MonthBegin(1)
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
    conn, params=(str(start), str(end)))
df["timestamp"] = pd.to_datetime(df["timestamp"])
v = df[CH].astype(float).values
n = len(v)
step = df["timestamp"].diff().dropna().median()

# 1. Light smoothing (~4h). Heavier smoothing blurs boundaries; lighter
# keeps them but lets the noise leak into segmentation. 4h is the
# empirical sweet spot for this benchmark.
W = max(4, int(round(pd.Timedelta(hours=4) / step)))
smooth = pd.Series(v).rolling(W, center=True).mean().bfill().ffill().values
min_gap = max(W * 4, int(round(pd.Timedelta(days=5) / step)))

# 2. Candidate boundaries: top |second derivative| positions.
d1 = np.diff(smooth, prepend=smooth[0])
d2 = np.abs(np.diff(d1, prepend=d1[0]))

def cuts_for_n(n_cuts):
    order = np.argsort(d2)[::-1]
    cuts = []
    for idx in order:
        if idx < min_gap or idx > n - min_gap - 1:
            continue
        if all(abs(idx - c) > min_gap for c in cuts):
            cuts.append(int(idx))
        if len(cuts) >= n_cuts:
            break
    return sorted(cuts)

def pl_sse(bounds):
    """Total SSE of best-fit lines on each segment."""
    s = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = smooth[a:b + 1]
        if len(seg) < 2:
            continue
        x = np.arange(len(seg))
        slope, intercept = np.polyfit(x, seg, 1)
        s += float(np.sum((seg - (slope * x + intercept)) ** 2))
    return s

# 3. Adaptive n_cuts via BIC. Try 1/2/3 and pick the lowest cost.
PENALTY = 8.0
best_cuts, best_score = [], float("inf")
for nc in (1, 2, 3):
    cuts = cuts_for_n(nc)
    bounds = [0] + cuts + [n - 1]
    score = pl_sse(bounds) + PENALTY * nc * np.log(n) * float(np.var(smooth))
    if score < best_score:
        best_score, best_cuts = score, cuts
cuts = best_cuts

# 4. Refine each boundary: slide ±REFINE_RADIUS to minimise total SSE.
REFINE_RADIUS = 8
def refine(cuts):
    cuts = list(cuts)
    if not cuts:
        return cuts
    for _ in range(3):                            # at most 3 rounds, usually converges
        changed = False
        for i in range(len(cuts)):
            lo = (cuts[i - 1] + 4) if i > 0 else 4
            hi = (cuts[i + 1] - 4) if i < len(cuts) - 1 else n - 5
            base_bounds = [0] + cuts + [n - 1]
            base = pl_sse(base_bounds)
            best_pos = cuts[i]
            for offset in range(-REFINE_RADIUS, REFINE_RADIUS + 1):
                if offset == 0:
                    continue
                new_pos = cuts[i] + offset
                if new_pos < lo or new_pos > hi:
                    continue
                trial = sorted(cuts[:i] + [new_pos] + cuts[i + 1:])
                s = pl_sse([0] + trial + [n - 1])
                if s < base:
                    base = s; best_pos = new_pos; changed = True
            cuts[i] = best_pos
        if not changed:
            break
    return cuts
cuts = refine(cuts)
bounds = [0] + cuts + [n - 1]

# 5. Per-segment slope + detrended-std.
ref_range = max(1e-9, float(np.ptp(smooth)))
seg_info = []
for a, b in zip(bounds[:-1], bounds[1:]):
    if b - a < 4:
        continue
    seg = smooth[a:b + 1]
    slope = float((seg[-1] - seg[0]) / max(1, len(seg) - 1))
    rel_slope = abs(slope) * (len(seg) - 1) / ref_range
    x = np.arange(len(seg))
    detrend = seg - (seg[0] + slope * x)
    rel_std = float(np.std(detrend)) / ref_range
    seg_info.append({"a": a, "b": b, "slope": slope,
                     "rel_slope": rel_slope, "rel_std": rel_std})

# 6. Classify KIND.
SLOPE_LOW = 0.15
for s in seg_info:
    if s["rel_slope"] < SLOPE_LOW:
        s["kind"] = "stable"
    else:
        s["kind"] = "rise" if s["slope"] > 0 else "fall"

# 7. Classify ADJ by within-month relative ranking.
RAPID_FRAC = 0.60
FLUCT_FRAC = 0.50

rfs = [s for s in seg_info if s["kind"] in ("rise", "fall")]
if rfs:
    mx = max(s["rel_slope"] for s in rfs)
    for s in rfs:
        s["adj"] = "rapid" if s["rel_slope"] >= mx * RAPID_FRAC else "gradual"

sts = [s for s in seg_info if s["kind"] == "stable"]
if sts:
    mx = max(s["rel_std"] for s in sts)
    for s in sts:
        s["adj"] = "fluctuating" if s["rel_std"] >= mx * FLUCT_FRAC else "steady"

trend_segments = [
    {"start": df["timestamp"].iloc[s["a"]].strftime("%Y-%m-%d %H:%M:%S"),
     "end":   df["timestamp"].iloc[s["b"]].strftime("%Y-%m-%d %H:%M:%S"),
     "adj":   s["adj"], "kind":  s["kind"]}
    for s in seg_info
]

# 8. Outlier: max |z-score residual| per enclosing segment.
resid = np.zeros_like(v)
for a, b in zip(bounds[:-1], bounds[1:]):
    seg = v[a:b + 1]
    mu = float(np.mean(seg))
    sd = float(np.std(seg)) or 1e-9
    resid[a:b + 1] = (seg - mu) / sd
idx_max = int(np.argmax(np.abs(resid)))
outliers = []
if abs(resid[idx_max]) > 3.0:
    outliers.append({
        "timestamp": df["timestamp"].iloc[idx_max].strftime("%Y-%m-%d %H:%M:%S"),
        "value": float(v[idx_max]),
    })

_result = {"trend_segments": trend_segments, "outliers": outliers}
```

## Rules

* **Return a DICT** (`trend_segments` + `outliers`).
* **Adaptive n_cuts** via BIC; **refine boundaries** by local SSE slide.
* **Adj is RELATIVE within the month** — never use an absolute slope threshold for rapid/gradual or std for fluctuating/steady.
* `adj ∈ {gradual, rapid, steady, fluctuating}`; `kind ∈ {rise, fall, stable}`.
* At most ONE significant outlier per month.
* Use raw_data only.
