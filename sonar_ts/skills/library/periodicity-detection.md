---
id: periodicity-detection
description: For "dominant cycle period (in data points)" questions — use autocorrelation (first peak after lag 0) as the primary estimator; cross-check with FFT but discard sub-harmonic candidates that are 2-3x larger than the autocorrelation peak.
---

# Periodicity Detection — autocorrelation first, FFT as verifier

## Why not just FFT.argmax?

The naïve `np.argmax(np.abs(np.fft.rfft(y)))` often returns a **harmonic or sub-harmonic** of the true period, not the fundamental. Failure mode we see in this benchmark: prediction = 2× or 3× the ground truth (the LLM picked a low-frequency bin where the harmonic happens to be slightly stronger than the fundamental, or low-frequency drift bled energy into near-DC bins).

The reliable estimator for "fundamental cycle period in samples" is the **first significant peak of the autocorrelation function past lag 0**.

## Algorithm

```python
df = pd.read_sql_query(
    'SELECT timestamp, "<ch>" FROM raw_data '
    "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
    conn, params=(search_start, search_end))
y = df["<ch>"].astype(float).values
y = y - y.mean()                              # remove DC
N = len(y)

# Normalised autocorrelation via numpy correlate (O(N²) but N is small
# for this task — typically a few hundred to a few thousand samples).
acf = np.correlate(y, y, mode="full")
acf = acf[N - 1:] / acf[N - 1]                # keep non-negative lags, normalise

# Find the first local maximum at lag >= 2. A "peak" satisfies
# acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.1.
period = None
for i in range(2, len(acf) - 1):
    if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
        period = i
        break

# Fallback: if no peak crosses the 0.1 threshold, use FFT argmax but
# restrict the period range to [3, N/3] to avoid trivial low-frequency
# bins. Then pick the argmax in that restricted band.
if period is None:
    mag = np.abs(np.fft.rfft(y))
    mag[0] = 0.0
    # FFT bin k corresponds to period N/k. We want periods in [3, N/3],
    # i.e. bins k in [3, N/3]. Equivalently, exclude bins < 3 and > N/3.
    lo, hi = 3, max(3, len(mag) // 1)
    # Mask out bins that would correspond to period > N/3 (very low freq).
    k_min = max(1, int(np.ceil(N / (N / 3))))   # = 3
    k_max = max(k_min + 1, int(N // 3))
    mask = np.zeros_like(mag)
    mask[k_min:k_max + 1] = mag[k_min:k_max + 1]
    k = int(np.argmax(mask))
    period = int(round(N / k))

# Final output: a single int (number of data points per cycle).
_result = int(period)
```

## Sanity check before emitting

If the autocorrelation method finds `period`, verify by checking that `acf[2 * period]` and `acf[3 * period]` are also positive (harmonics support the fundamental). If they are NEGATIVE, the algorithm probably picked a half-period — try doubling and re-check.

```python
if 2 * period < len(acf) and acf[2 * period] < -0.1:
    # Picked a half-period; the true period is 2x.
    period = 2 * period
```

## Common pitfalls

* **Don't use raw FFT argmax** without restricting the period band — the largest spectral bin is often very-low-frequency drift, not the cycle.
* **Don't return 0 or 1** — those are degenerate; the benchmark periods are typically in [4, 200] samples.
* **Output type must be int**, not numpy.int64. Use `int(period)` (the formatter handles the rest, but `int()` is safer than `.item()`).
* **Use raw_data only**; the feature tables aggregate at 1-day / 1-month granularity which is unrelated to sub-window periodicity.
