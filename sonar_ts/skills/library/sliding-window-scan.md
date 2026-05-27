---
id: sliding-window-scan
description: For "which K-day period in <year> had the {highest/lowest} {metric}" — scan raw_data with a K-day window at the data's native sampling resolution.
---

# Sliding-window scan over raw_data

## Why raw_data and not the feature tables

`daily_feature` / `monthly_feature` / `yearly_feature` only index windows of length exactly 1 day / 1 month / 1 year. They CANNOT answer arbitrary K-day queries. Querying e.g. `daily_feature.avg_val ORDER BY avg_val ASC LIMIT 1` returns the lowest-average single day, not the lowest-average K-day window — your reported interval will be 1 day wide and score ~0. Always work on raw_data for this task.

## Algorithm (vectorised, ~0.3 s on a year of 15-min data)

```python
df = pd.read_sql_query(
    'SELECT timestamp, "<ch>" FROM raw_data '
    "WHERE timestamp >= '<year>-01-01' AND timestamp < '<year+1>-01-01' "
    "ORDER BY timestamp", conn)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert K days to a sample count using the native sampling step.
step = df["timestamp"].diff().dropna().median()
n_per_window = int(round(pd.Timedelta(days=K) / step))

s = df["<ch>"].astype(float).values        # plain numpy is faster + cleaner

# Vectorised rolling metric — pick the right one for the question:
import pandas as _pd
ser = _pd.Series(s)
if metric == "largest range":
    metric_arr = (ser.rolling(n_per_window).max()
                - ser.rolling(n_per_window).min()).values
elif metric in ("highest average", "lowest average"):
    metric_arr = ser.rolling(n_per_window).mean().values
elif metric in ("highest variance", "highest std"):
    metric_arr = ser.rolling(n_per_window).var().values
# ... extend as needed

# CRITICAL: rolling leaves the first n_per_window-1 entries as NaN. Use
# nanargmax / nanargmin — plain np.argmax treats NaN as +inf and returns
# index 0, then best_start_pos goes negative and wraps to the end of the
# year. This is the #1 source of zero-IoU on this task.
if metric.startswith("lowest"):
    best_end_pos = int(np.nanargmin(metric_arr))
else:
    best_end_pos = int(np.nanargmax(metric_arr))

best_start_pos = best_end_pos - n_per_window + 1

start = df["timestamp"].iloc[best_start_pos].strftime("%Y-%m-%d %H:%M:%S")
end   = df["timestamp"].iloc[best_end_pos  ].strftime("%Y-%m-%d %H:%M:%S")
_result = [start, end]
```

## Self-check before returning

After picking `_result`, verify the interval width matches what was asked:

```python
delta = pd.Timestamp(end) - pd.Timestamp(start)
assert abs(delta - pd.Timedelta(days=K)) < pd.Timedelta(hours=1), \
    f"window width {delta} ≠ {K} days — bug in indexing"
```

If this fails, the most common cause is forgetting `nanargmax` (see above).
