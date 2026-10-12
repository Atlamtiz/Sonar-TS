---
id: dominant-cycle-period-estimation-s6
description: Use when the question asks for the dominant cycle period (in data points) of a single channel within a given time range — the length of the repeating pattern in the series — and requires a robust period estimate from a noisy, possibly non-stationary signal. The answer is a single integer period in samples, not a timestamp, range, or list. Not for trend segmentation, ramp/step detection, window-length optimisation, or plain aggregates.
---

# Robust dominant-cycle period estimation (detrend → FFT primary → ACF disambiguation → fold-verify)

## When to use
The question names one channel, one time range, and asks for the dominant cycle period in data points. The answer is one integer. It is NOT for trend segmentation, ramp/step detection, window-length optimisation, or plain aggregates.

## Algorithm
1. Read the raw series for `<channel>` over `[<start>, <end>)` ordered by timestamp. Drop non-finite values (`x = x[np.isfinite(x)]`), then `n = len(x)`. The answer is in DATA POINTS, so work in sample index, not time.
2. **Detrend** with a window strictly longer than the longest plausible period: `win = max(5, 2*p_max + 1)` where `p_max = n // 2`. A shorter window (e.g. `n//10`) removes the cycle itself. Subtract the rolling median.
3. **Plausible range**: `p_min = 3`, `p_max = max(p_min, n // 2)`. If `n < 6`, still run the ACF with `p_max = n - 1` rather than fabricating a value.
4. **FFT primary**: Hann-window the detrended series, take `|rfft|`, restrict to periods in `[p_min, p_max]`, take the peak period `p_fft`. Clip `p_fft = min(max(p_fft, p_min), p_max)`.
5. **ACF disambiguation**: compute the ACF via FFT. Take the **global argmax of the ACF restricted to `[p_min, p_max]`** as `p_acf` (this is the correct rule for noisy data — the "decay-then-first-local-max" rule fires on 1–2 sample wiggles). If `p_fft` is within ±20% of `p_acf`, keep `p_acf`; otherwise keep `p_fft` (FFT is more robust to non-stationarity).
6. **Fold-verify with a normalised coherence**: for each candidate `p` in `{p_fft, p_acf, round(p_acf/2), 2*p_acf}` clipped to `[p_min, p_max]`, fold the detrended series at period `p` (phase = index mod p), compute the phase-mean profile, and score `coh_norm = (prof.var() / (y.var() + 1e-12)) * (p - 1) / (n - 1)`. The `(p-1)/(n-1)` factor corrects the small-`p` bias (a small period always fits noise better). Highest `coh_norm` wins; ties broken by larger `p` (prefer the fundamental over a harmonic).
7. Clip the winner: `best_p = min(max(best_p, p_min), p_max)`. Return `int(round(best_p))`.

```python
import numpy as np, pandas as pd

CH = "<channel>"
START = "<start>"
END = "<end>"

df = pd.read_sql_query(
    f'SELECT timestamp, "{CH}" AS v FROM raw_data '
    'WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp',
    conn, params=(START, END))
x = df["v"].to_numpy(dtype=float)
x = x[np.isfinite(x)]
n = len(x)
if n < 6:
    # too short for a cycle; still run ACF with p_max = n-1
    p_min, p_max = 3, max(3, n - 1)
else:
    p_min, p_max = 3, max(3, n // 2)

# detrend with a window strictly longer than p_max
win = max(5, 2 * p_max + 1)
trend = pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()
y = x - trend
y = y - y.mean()
sd = y.std()
if sd == 0 or not np.isfinite(sd):
    _result = int(max(1, n // 2))
else:
    y = y / sd

    # FFT primary
    w = np.hanning(n)
    Y = np.abs(np.fft.rfft(y * w))
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= p_min) & (periods <= p_max)
    p_fft = int(round(periods[mask][np.argmax(Y[mask])])) if mask.any() else p_min
    p_fft = min(max(p_fft, p_min), p_max)

    # ACF via FFT
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(y, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    acf = acf / acf[0]
    hi = min(p_max, n - 1)
    lo = p_min
    if hi <= lo:
        lo, hi = p_min, max(p_min, n - 1)
    seg = acf[lo:hi + 1]
    p_acf = lo + int(np.argmax(seg))

    # combine
    if abs(p_fft - p_acf) <= 0.2 * max(p_fft, p_acf):
        best_p = p_acf
    else:
        best_p = p_fft

    # fold-verify with normalised coherence
    cands = {p_fft, p_acf, max(p_min, p_acf // 2), min(p_max, 2 * p_acf)}
    best_coh = -1.0
    for p in sorted(cands):
        if p < p_min or p > p_max:
            continue
        phase = np.arange(n) % p
        prof = np.zeros(p)
        cnt = np.zeros(p)
        np.add.at(prof, phase, y)
        np.add.at(cnt, phase, 1.0)
        cnt[cnt == 0] = 1.0
        prof = prof / cnt
        coh = (prof.var() / (y.var() + 1e-12)) * (p - 1) / (n - 1)
        if coh > best_coh + 1e-9 or (abs(coh - best_coh) <= 1e-9 and p > best_p):
            best_coh, best_p = coh, p

    best_p = min(max(best_p, p_min), p_max)
    _result = int(round(best_p))
```

## Rules
- Answer is in DATA POINTS (sample index), never in seconds — do not convert via `dt`.
- Always detrend before ACF/FFT; the detrending window must be strictly longer than `p_max` (`win = max(5, 2*p_max + 1)`), otherwise the cycle itself is removed.
- Restrict the search to `[3, n//2]`; a period of 1–2 samples is not a cycle. Clip the final answer to this range.
- FFT is the primary estimator; ACF is used only to disambiguate harmonics. Take the **global argmax of the ACF restricted to `[p_min, p_max]`** — do NOT use a "first local maximum after decay" rule, which fires on 1–2 sample wiggles in noisy data.
- Verify by folding at the period and score with the **normalised** coherence `coh * (p-1)/(n-1)` so candidates of different `p` are comparable; prefer the fundamental over a harmonic on ties.
- Drop non-finite values before any processing; recompute `n`.
- Return an integer; round to the nearest sample.
- If the series is constant, return a safe fallback (`max(1, n//2)`).

## Pitfalls
- Raw FFT peak lands on a high-frequency bin (period 1–2) or a low-frequency bin (period ≈ n) — detrend and restrict the range.
- ACF "first local maximum after decay" rule fires on 1–2 sample wiggles in noisy data — use the global argmax over `[p_min, p_max]` instead.
- Detrending window shorter than `p_max` removes the cycle itself — use `win = max(5, 2*p_max + 1)`.
- Unnormalised fold-coherence is biased toward small `p` — multiply by `(p-1)/(n-1)`.
- Forgetting to mean-remove before ACF/FFT — the DC component swamps the spectrum.
- Returning a harmonic (p/2) instead of the fundamental — the fold-coherence step with the "prefer larger p on tie" rule fixes this.
- Non-integer output — always `int(round(...))`.
- NaNs in the raw channel propagate silently — filter with `np.isfinite` first.
- Empty candidate set (very short series) — fall back rather than crash.
