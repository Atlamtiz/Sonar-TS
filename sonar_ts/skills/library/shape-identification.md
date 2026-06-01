---
id: shape-identification
description: For "identify the time range of the {superlative} {shape} in channel X within window W" — Search-Then-Verify with SAX. SEARCH writes a SAX regex over daily_feature to find which days in the window carry the shape; VERIFY runs a shape-specific detector on the raw data of the candidate span (first→last candidate day). If the SAX search returns nothing, fall back to the whole window.
---

# Shape Identification — SAX Search-Then-Verify

`SQL SAX regex → candidate days → (empty? fall back to whole window) → verify the candidate span → interval`.

`daily_feature.sax` is a 24-char string per (channel, day), alphabet `a..e` over 5 equiprobable
z-normalised bands. A regex on SAX approximates the shape (paper's canonical example:
`WHERE regexp_like(sax, '[ab]+.*[de]+.*[ab]+')` for rise-then-fall). The regex is fuzzy — it
says which days carry the shape; the Python detector nails the exact boundary on the raw signal
of the candidate span. The shape may cross day boundaries, so verify a contiguous span, not
single days. Use the code below almost verbatim.

```python
import re
import numpy as np, pandas as pd
from datetime import timedelta

CH = "<channel id>"
SHAPE = "<plateau|depression|spike|valley|step_up|step_down>"
SEARCH_START = "<YYYY-MM-DD HH:MM:SS>"
SEARCH_END   = "<YYYY-MM-DD HH:MM:SS>"

# ---------- SEARCH: SAX regex over the feature index (plain SAX, paper §5.3 / Fig.4) ----------
REGEX_BY_SHAPE = {                     # recall-first fuzzy bands (the regex only confirms
    "plateau":    r"(.)\1{2,}",        # which days carry the shape; the detector fixes bounds)
    "depression": r"[ab]{2,}",         # low band present
    "spike":      r"[de]+",            # reaches a high band
    "valley":     r"[ab]+",            # reaches a low band
    "step_up":    r"[a-c].*[c-e]",     # low region -> high region
    "step_down":  r"[c-e].*[a-c]",     # high region -> low region
}
hits = pd.read_sql_query(
    "SELECT window_start FROM daily_feature "
    "WHERE channel_id = ? AND window_start >= ? AND window_start <= ? AND sax REGEXP ?",
    conn, params=(CH, SEARCH_START[:10] + " 00:00:00", SEARCH_END,
                  REGEX_BY_SHAPE.get(SHAPE, r".*")))
cand_days = sorted({pd.Timestamp(s).date() for s in hits["window_start"]})

# ---------- VERIFY span: candidate span, or whole window if SAX empty (Fig.4 fallback) ----------
if cand_days:
    span_lo = max(f"{cand_days[0]} 00:00:00", SEARCH_START)
    span_hi = min(f"{cand_days[-1] + timedelta(days=1)} 23:59:59", SEARCH_END)
else:
    span_lo, span_hi = SEARCH_START, SEARCH_END
df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" FROM raw_data '
    "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
    conn, params=(span_lo, span_hi))
df["timestamp"] = pd.to_datetime(df["timestamp"])
v = df[CH].astype(float).values

def _longest_true(mask):
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

n = len(v)
if SHAPE in ("plateau", "depression"):
    rstd = pd.Series(v).rolling(4, center=True).std().bfill().ffill().values
    mask = rstd <= 0.10 * float(np.std(v))
    if SHAPE == "depression":
        mask &= (v <= np.percentile(v, 30))
    r = _longest_true(mask); a, b = (r if r else (0, n - 1))
elif SHAPE in ("spike", "valley"):
    idx = int(np.argmax(v)) if SHAPE == "spike" else int(np.argmin(v))
    mid = ((np.max(v) + np.median(v)) / 2 if SHAPE == "spike"
           else (np.min(v) + np.median(v)) / 2)
    a = idx
    while a > 0 and ((v[a-1] >= mid) if SHAPE == "spike" else (v[a-1] <= mid)):
        a -= 1
    b = idx
    while b < n-1 and ((v[b+1] >= mid) if SHAPE == "spike" else (v[b+1] <= mid)):
        b += 1
else:  # step_up / step_down: largest monotone transition (cross-day OK), O(n)
    W = max(8, n // 24)
    sm = pd.Series(v).rolling(W, center=True).mean().bfill().ffill().values
    if SHAPE == "step_up":
        b = int(np.argmax(sm - np.minimum.accumulate(sm)))
        a = int(np.argmin(sm[:b+1])) if b > 0 else 0
    else:
        b = int(np.argmax(np.maximum.accumulate(sm) - sm))
        a = int(np.argmax(sm[:b+1])) if b > 0 else 0

_result = [df["timestamp"].iloc[a].strftime("%Y-%m-%d %H:%M:%S"),
           df["timestamp"].iloc[b].strftime("%Y-%m-%d %H:%M:%S")]
```

## Rules
* **Search-Then-Verify, strict**: candidates come from the SQL `sax REGEXP` search; verify the raw data of the candidate span (first→last candidate day, +1) — not the whole window blindly. Only when the SAX search is empty do you fall back to the full window (paper's Fig.4 rule).
* Recall-first regex (it confirms which days carry the shape; the detector fixes the exact boundary). `sax REGEXP` / `regexp_like` both registered. No external knowledge of the answer.
* Use the code above almost verbatim. Never call `exit()` / `sys.exit()`.
