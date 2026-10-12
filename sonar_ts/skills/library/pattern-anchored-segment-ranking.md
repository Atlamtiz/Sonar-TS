---
id: pattern-anchored-segment-ranking
description: Use when the question asks for days whose hourly curve matches a described ordered sequence of trend parts (e.g. 'slow fall, then rapid rise') and then ranks those days by how slow, fast, steep or fluctuating one named part is (e.g. the slowest fall, the fastest rise), returning the top-k dates; the named part is located at the token positions of the matched symbolic pattern and the property measured on raw_data over exactly those hours. Also covers filter-then-rank questions. Not for questions that only filter by a pattern without ranking, nor for ranking whole windows by an aggregate, nor for questions that name the ranked segment as 'rapid rise' or 'rapid fall' and ask which day's such segment is the fastest.
---

# Pattern-anchored segment ranking

## When to use
The question names an ordered sequence of trend parts for a curve and asks for the top-k windows ranked by a property of ONE named part (slowest fall, fastest rise, most fluctuating stretch). The named part is located by the token positions of the matched `sym` pattern — never by a fresh heuristic on raw_data. Works for pure ranking and for filter-then-rank questions.

## Algorithm
1. **Read placeholders**: `<channel>`, `<start>`, `<end>`, `<k>`, the ordered parts `<pattern>`, and which part is RANKED with its property. The ranked part is the one the ranking clause names; its position in the ordered parts list determines which group is the capture. Property → key, measured on the raw slice of the ranked part: `slowest`/`most gradual` → smallest `|slope|`; `fastest` → largest signed slope; `steepest` → largest `|slope|`; `most fluctuating` → largest `std`; `longest` → largest duration.
2. **Choose the feature table by the DURATION OF THE SHAPE ITSELF**, not the span of the question's time range: hours → `daily_feature` (1 token = 1 hour), days → `monthly_feature`, months → `yearly_feature`.
3. **Build the strict regex** from the parts in order, using the codebook classes, joined by the whole-token skip `(?:[a-e][RFS][012][=~])*?`. Examples: `slow fall` → `(?:[a-e]F1[=~])+`; `rapid rise` → `(?:[a-e]R2[=~])+`; `rise` → `(?:[a-e]R[012][=~])+`; `fluctuating stable` → `(?:[a-e][RFS][012]~){2,}`; `steady stable` → `(?:[a-e]S0=){2,}`. Wrap the RANKED part in a **capturing group** in BOTH strict and loose patterns.
4. **Search the chosen table** filtered by `channel_id` and `window_start` in `[<start>, <end>)`. If fewer than `<k>` rows return, run the loose pattern and take the union.
5. **Loosening rule — widen the trend slot, not just the rate slot.** A slow descent is often expressed as a run of `S` tokens whose level letters descend (`dS0= cS0= bS0= aS0=`), not as `F1` tokens. For a ranked FALL part, the loose pattern must widen BOTH the trend slot to `[RFS]` AND the rate slot to `[012]`, keeping the level slot `[a-e]` and the stability slot `[=~]`. For a ranked RISE part, widen the trend slot to `[RFS]` and the rate slot to `[012]` likewise. For a ranked STABLE/fluctuating part, widen the stability slot to `[=~]` and the trend slot to `[RFS]`. Never drop the level slot. If the loose search is still short, fall back to raw_data.
6. **Locate the ranked segment per candidate** by `re.search(pattern, sym)` in Python and reading `m.start(1)` / `m.end(1)`. Token index = char position // 4; hour offset = token index (for `daily_feature`). `tok_b` is the EXCLUSIVE token index of the ranked part's end. Require `tok_b - tok_a >= 2`. Skip candidates whose `sym_len < 24` (incomplete first/last window). If the pattern has no group, skip the candidate.
7. **Verify on raw_data** for the shortlisted windows only: fetch the window's hours, slice `[h0, h1)` where `h0 = ws + tok_a hours` and `h1 = ws + tok_b hours` (exclusive end — do NOT add an extra hour). Compute the ranking quantity on that slice: `slope = (v[-1] - v[0]) / (t[-1] - t[0])` in units per hour for a fall/rise; `np.std(v)` for fluctuation. For a ranked FALL, drop candidates whose raw slope is not negative; for a ranked RISE, drop candidates whose raw slope is not positive. If the slice has fewer than 2 points, skip the window.
8. **Rank** by the key: `abs(slope)` ascending for slowest fall/rise; signed slope descending for fastest rise; `abs(slope)` descending for fastest fall; `std` descending for most fluctuating. Deduplicate by date BEFORE ranking, keeping the best key per date. Tie-break by earlier `window_start`. Return the top-`<k>` dates as `YYYY-MM-DD` strings.

```python
import re, numpy as np, pandas as pd

CH = "<channel>"
START = "<start>"; END = "<end>"
K = <k>
TABLE = "daily_feature"          # chosen by shape duration
TOK_HOURS = 1                    # 1 for daily, 24 for monthly, 720 for yearly
STRICT = r"<strict_pattern_with_capture_on_ranked_part>"
LOOSE  = r"<loose_pattern_with_capture_on_ranked_part>"
PROPERTY = "<slowest|fastest|steepest|most_fluctuating>"
CAP = max(400, 20 * K)

q = f"""SELECT window_start, window_end, sym, sym_len FROM {TABLE}
        WHERE channel_id = ? AND window_start >= ? AND window_start < ?
          AND sym REGEXP ?"""
cands = pd.read_sql_query(q, conn, params=(CH, START, END, STRICT))
if len(cands) < K:
    extra = pd.read_sql_query(q, conn, params=(CH, START, END, LOOSE))
    cands = pd.concat([cands, extra]).drop_duplicates("window_start")
cands = cands.head(CAP)

spans = []
for _, r in cands.iterrows():
    if int(r["sym_len"]) < 24:
        continue
    m = re.search(STRICT, r["sym"]) or re.search(LOOSE, r["sym"])
    if not m or m.re.groups < 1 or m.group(1) is None:
        continue
    tok_a, tok_b = m.start(1) // 4, m.end(1) // 4
    if tok_b - tok_a < 2:
        continue
    spans.append((pd.Timestamp(r["window_start"]), tok_a, tok_b))

if spans:
    lo = min(s[0] for s in spans).strftime("%Y-%m-%d %H:%M:%S")
    hi = (max(s[0] for s in spans) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    raw = pd.read_sql_query(
        f'SELECT timestamp, "{CH}" AS v FROM raw_data '
        'WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp',
        conn, params=(lo, hi))
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index("timestamp")["v"].astype(float)

rows = []
for ws, tok_a, tok_b in spans:
    h0 = ws + pd.Timedelta(hours=tok_a * TOK_HOURS)
    h1 = ws + pd.Timedelta(hours=tok_b * TOK_HOURS)   # exclusive
    seg = raw.loc[(raw.index >= h0) & (raw.index < h1)]
    if len(seg) < 2:
        continue
    v = seg.values
    t = seg.index.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    if t[-1] <= t[0]:
        continue
    if PROPERTY == "most_fluctuating":
        key = float(np.std(v))
    else:
        slope = (v[-1] - v[0]) / (t[-1] - t[0])
        if PROPERTY in ("slowest", "most_gradual"):
            if slope >= 0:      # ranked fall must actually fall
                continue
            key = abs(slope)
        elif PROPERTY == "fastest":
            key = slope
        else:                   # steepest
            key = abs(slope)
    rows.append((ws.strftime("%Y-%m-%d"), key))

# dedup by date BEFORE ranking, keep best key per date
best = {}
for d, k in rows:
    if d not in best:
        best[d] = k
    elif PROPERTY in ("slowest", "most_gradual"):
        best[d] = min(best[d], k)
    else:
        best[d] = max(best[d], k)

if PROPERTY in ("slowest", "most_gradual"):
    ranked = sorted(best.items(), key=lambda x: (x[1], x[0]))
else:
    ranked = sorted(best.items(), key=lambda x: (-x[1], x[0]))

_result = [d for d, _ in ranked[:K]]
```

## Rules
- The ranked segment's span comes from the **token positions of the matched sym** (`m.start(1)//4`, `m.end(1)//4`); the ranking **quantity** is measured on raw_data over exactly those hours. Never re-derive the span from a fresh argmax/argmin or longest-monotone-run scan.
- The capture group must sit on the part the ranking clause names, in BOTH strict and loose patterns; skip candidates whose match has no group.
- Choose the feature table by the shape's own duration (hours → daily, days → monthly, months → yearly), never by the question's time span.
- **Loosening widens the trend slot too.** A slow fall can read as `S` tokens with descending level letters, so the loose pattern for a ranked fall must accept `[RFS]` in the trend slot and `[012]` in the rate slot; the level slot stays `[a-e]`.
- For a ranked fall, drop candidates whose raw slope is not negative; for a ranked rise, drop candidates whose raw slope is not positive.
- Require `tok_b - tok_a >= 2` and `sym_len >= 24`; skip incomplete windows.
- The slice end is exclusive: `h1 = ws + tok_b * TOK_HOURS`, no `+1h`.
- Deduplicate by date BEFORE ranking, keeping the best key per date; tie-break by earlier `window_start`.
- Candidate cap is `max(400, 20*K)`.
- Convert timestamps to `YYYY-MM-DD` strings before assigning `_result`.

## Pitfalls
- Anchoring the capture group to the first F-run in the string instead of the fall that sits between the preceding and following named parts — days where the fall occurs later in the day get mis-located.
- Loosening only the rate slot while keeping the trend letter `F`, so days whose slow fall reads as `S` tokens with descending levels never enter the shortlist.
- Trusting the sym match without verifying the named sequence on raw_data — a brief dip inside a rise can coincidentally match.
- Ranking by duration or drop-per-step of an arbitrary run instead of the slope of the pattern-anchored segment.
- Deduplicating by date AFTER sorting by key, which can drop a higher-ranked day in favour of a lower-ranked one.
- Forgetting to map token index → time offset (token i covers segment i of the window), so the measured slice is misaligned.
- Adding `+1h` to the slice end (double-counts one hour) or extending a short slice — skip such windows instead.
- Fetching raw_data per candidate inside a Python loop — fetch all shortlisted windows in one query and slice in memory.
- Empty candidate set or slices shorter than 2 points: skip those windows rather than crashing.
