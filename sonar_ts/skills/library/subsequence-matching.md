---
id: subsequence-matching
description: For "find the time interval where channel X exhibits the most similar pattern to the reference window" — slide a window of the SAME length as the query across the search range, score each position with z-normalised Euclidean (or DTW) distance, and return the lowest-distance window.
---

# Subsequence matching — match the query LENGTH exactly

The question gives two ranges:
* `query_window` — the reference pattern (typically 8-12 hours wide).
* `search_window` — the haystack where the match must live (~5-7 days).

The answer interval **must have the same number of samples as the query**. Predictions wider/narrower than the query score zero IoU. The most common mistake is returning a fixed 1-day window regardless of the query length.

## Algorithm

```python
# 1. Pull both segments from raw_data.
def fetch(start, end):
    df = pd.read_sql_query(
        'SELECT timestamp, "<ch>" FROM raw_data '
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        conn, params=(start, end))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

qdf = fetch(query_start, query_end)
sdf = fetch(search_start, search_end)
q = qdf["<ch>"].astype(float).values
s = sdf["<ch>"].astype(float).values
L = len(q)                 # window length — fixed by the query

# 2. Z-normalise both signals so we compare shape, not absolute level.
def znorm(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / sd if sd > 1e-9 else x - mu

qz = znorm(q)

# 3. Slide a window of length L across s, compute Euclidean distance
# between qz and z-normalised window. Vectorise the rolling mean/std for
# speed — O(n) instead of O(n*L).
ser = pd.Series(s)
roll_mean = ser.rolling(L).mean().shift(-(L - 1)).values   # left-aligned
roll_std  = ser.rolling(L).std().shift(-(L - 1)).values
# (Convert right-aligned rolling result to left-aligned by shifting.)

best_dist = np.inf
best_i = 0
for i in range(len(s) - L + 1):
    mu, sd = roll_mean[i], roll_std[i]
    if not np.isfinite(mu) or sd < 1e-9:
        continue
    wz = (s[i:i + L] - mu) / sd
    dist = float(np.linalg.norm(qz - wz))
    if dist < best_dist:
        best_dist = dist
        best_i = i

# 4. Emit timestamps from the search-window DataFrame.
start = sdf["timestamp"].iloc[best_i].strftime("%Y-%m-%d %H:%M:%S")
end   = sdf["timestamp"].iloc[best_i + L - 1].strftime("%Y-%m-%d %H:%M:%S")
_result = [start, end]
```

## When DTW helps

If the prototype is described as warped (e.g. "stretched", "compressed", "variable speed"), Euclidean can miss it. Replace step 3's distance with DTW (use `scipy.spatial.distance` if available, otherwise hand-roll the classic O(L²) recurrence). For straightforward prototypes (bell curve, double-peak, step pattern, sharp spike — all the prototypes in this benchmark), z-normalised Euclidean is already excellent.

## Rules

* **Output length must equal query length** in samples — sanity-check `(best_i + L - 1) - best_i + 1 == L` before emitting.
* **Stick to raw_data**; feature-table windows are 1-day granularity and cannot match arbitrary lengths.
* **Use `str(ts)[:19]` or `.strftime(...)`** to stringify timestamps for json.dumps.
* Don't pre-downsample; the GT boundaries are at native sample resolution.
