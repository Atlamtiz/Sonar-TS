"""MOSO — MOrphological SymbOlic representation (written from the definition, 2026-09-22).

One symbolic string per index window, exactly on the windows and segments of the Sonar-TS
feature tables:

    daily_feature    window = calendar day     segment = 1 hour
    monthly_feature  window = calendar month   segment = 1 day
    yearly_feature   window = calendar year    segment = 1 month

Per segment we fit the line  y = a + b*t + eps  to the raw points of that segment and
quantise its complete statistics (level, sign of b, size of b, size of eps) into a 4-tuple

    <Amplitude, Trend, Rate, Stability>

written as 4 characters, one character class per slot:

    Amplitude   a b c d e   the SAX letter of the segment mean (z-score inside the window,
                            Gaussian equiprobable breakpoints) — identical to plain SAX
    Trend       R F S       rise / fall / stable
    Rate        0 1 2       0 = not applicable (stable), 1 = gradual, 2 = rapid
    Stability   = ~         = steady, ~ fluctuating

THRESHOLDS — the DEPLOYED policy ("local", frozen 2026-09-22 on TRAIN + DEV data only):
Words of degree compare the parts of ONE window with each other; the channel's history only
answers whether anything happens at all.
    moving      |b| >= max( median |b| of the window's segments,
                            60 % quantile of |b| over the channel's history in this table )
    stable      otherwise (trend S, rate 0)
    rapid / slow  among the MOVING segments of the window: |b| >= their median -> 2, else 1
    fluctuating sigma >= max( median sigma of the window, 80 % quantile of sigma over the channel )
Every within-window cut is a median (the natural two-class split, no chosen number); the two
channel-level gate levels (0.60 for movement, 0.80 for scatter) were selected on the 200
TRAIN + DEV composite-trend tasks (DESIGN_VARIANTS*.md: recall is flat for gate levels 0.5–0.7 and
0.5–0.8, lift rises with them; 0.8 for movement starts to cost recall). Only raw values are
used, never a label; the threshold VALUES follow from each channel's and window's own data.

Design checks (literal word->pattern search, recall of injected days / candidate fraction / lift):
    TRAIN+DEV (200 CT)   SAX 0.51/0.26/2.0 (train)   MOSO mixed (previous) 0.852/0.140/6.1
                         MOSO local (deployed) 0.978/0.177/5.5
Ablations kept in the code: policy="mixed" (channel-level rapid/fluct quantiles, FROZEN_QUANTILES),
policy="jenks" (window natural breaks), rule="equi", rule="jenks-log", scope="channel", k>0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# table name -> (window frequency, segment frequency); the same rules as sonar_ts/offline.py
TABLES: Dict[str, Tuple[str, str]] = {
    "daily_feature": ("D", "h"),
    "monthly_feature": ("MS", "D"),
    "yearly_feature": ("YS", "MS"),
}
ALPHABET = "abcde"
_BREAKPOINTS = norm.ppf(np.linspace(0.0, 1.0, len(ALPHABET) + 1)[1:-1])


# --------------------------------------------------------------------------- segment statistics
_PERIOD = {"D": "D", "MS": "M", "YS": "Y"}


def segment_stats(series: pd.Series, seg_freq: str, win_freq: str | None = None, k: int = 0) -> pd.DataFrame:
    """Per segment: n, mean, slope b and residual standard deviation sigma of a least-squares line.

    The line of segment i is fitted to the raw points of segments i-k .. i+k (clipped at the
    window border, so a window's string depends on that window only). k = 0 uses the segment
    alone. b is the fitted change per segment (data units per hour / day / month); sigma is
    what the line leaves unexplained. Closed form, vectorised through rolling sums of the
    sufficient statistics."""
    s = series.dropna().sort_index()
    if seg_freq == "MS":
        seg = s.index.to_period("M").to_timestamp()
        dur = (seg + pd.offsets.MonthBegin(1)) - seg
    else:
        seg = s.index.floor(seg_freq)
        dur = pd.to_timedelta(np.full(len(s), 1), unit=seg_freq.lower())
    frac = ((s.index - seg) / dur).to_numpy(dtype=float)           # position inside the segment, 0..1
    y = s.to_numpy(dtype=float)
    own = pd.DataFrame({"seg": seg, "y": y}).groupby("seg", sort=True)["y"].agg(["size", "mean", "min", "max"])
    segs = own.index
    win = segs.to_period(_PERIOD[win_freq]).to_timestamp() if win_freq else pd.DatetimeIndex(np.full(len(segs), segs[0]))
    order = pd.Series(np.arange(len(segs)), index=segs)
    first = order.groupby(win).transform("min")
    x = (order - first).reindex(seg).to_numpy(dtype=float) + frac    # time in segment units since the window start
    fit = pd.DataFrame({"seg": seg, "x": x, "y": y})
    if k > 0:                                                        # context of neighbouring segments (same window)
        segw = pd.Series(win.to_numpy(), index=segs)
        parts = []
        for off in range(-k, k + 1):
            sh = fit.copy()
            pos = order.reindex(sh["seg"]).to_numpy() + off
            ok = (pos >= 0) & (pos < len(segs))
            sh = sh[ok]; pos = pos[ok]
            tgt = segs[pos]
            same = (segw.reindex(tgt).to_numpy() == segw.reindex(sh["seg"]).to_numpy())
            sh = sh[same].assign(seg=tgt[same])
            parts.append(sh)
        fit = pd.concat(parts)
    g = fit.groupby("seg", sort=True)
    n = g.size().astype(float)
    xm, ym = g["x"].transform("mean"), g["y"].transform("mean")   # centred form: no cancellation on constant segments
    dx, dy = fit["x"] - xm, fit["y"] - ym
    sxx = (dx * dx).groupby(fit["seg"]).sum()
    sxy = (dx * dy).groupby(fit["seg"]).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(sxx > 1e-12, sxy / sxx, 0.0)
    resid = dy - pd.Series(slope, index=sxx.index).reindex(fit["seg"]).to_numpy() * dx
    ssr = (resid * resid).groupby(fit["seg"]).sum()
    sigma = np.where(n >= 3, np.sqrt(ssr / np.maximum(n - 2, 1)), 0.0)
    out = pd.DataFrame({"n": own["size"].astype(float), "mean": own["mean"], "b": slope, "sigma": sigma}, index=segs)
    return clean_rounding_noise(out, series, constant=(own["max"] == own["min"]).to_numpy() if k == 0 else None)


def clean_rounding_noise(stats: pd.DataFrame, series: pd.Series, constant=None) -> pd.DataFrame:
    """Least squares on a constant segment returns |b| ~ 1e-13 instead of 0. Natural breaks
    inside a window would happily split that dust into flat / gradual / rapid, so it is removed:
    (a) a segment whose raw values are all equal has b = sigma = 0 exactly; (b) a fitted change
    or a residual below 1e-9 of the channel's value range is floating-point noise, not data.
    This is a numerical tolerance (like a machine epsilon), not a semantic threshold."""
    vals = series.dropna().to_numpy(dtype=float)
    tol = 1e-9 * float(vals.max() - vals.min()) if vals.size else 0.0
    b, s = stats["b"].to_numpy().copy(), stats["sigma"].to_numpy().copy()
    if constant is not None:
        b[constant], s[constant] = 0.0, 0.0
    b[np.abs(b) <= tol] = 0.0
    s[s <= tol] = 0.0
    if tol > 0:                          # snap to a grid of the tolerance: equal-within-noise -> exactly equal
        b = np.round(b / tol) * tol
        s = np.round(s / tol) * tol
    stats["b"], stats["sigma"] = b, s
    return stats


# --------------------------------------------------------------------------- natural breaks
def natural_breaks(values: np.ndarray, k: int, max_points: int = 2048) -> np.ndarray:
    """Fisher-Jenks natural breaks: the k-class partition of 1-D data with the smallest total
    within-class variance (exact dynamic programme). Returns the k-1 cut points, each the
    midpoint between the last value of a class and the first value of the next.

    Long inputs are represented by `max_points` evenly spaced order statistics (the empirical
    quantile function on a fixed grid) — a resolution setting like the 256 grey levels of
    Otsu's threshold, not a tuned parameter. With fewer than k distinct values every distinct
    value is its own class, filled from the lowest class upwards (one value -> all lowest)."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size > max_points:
        v = v[np.linspace(0, v.size - 1, max_points).round().astype(int)]
    # The programme runs over the DISTINCT values with their counts, so a cut can only fall
    # between two different values — never inside a run of equal ones (e.g. many exact zeros).
    u, w = np.unique(v, return_counts=True)
    if u.size < k:                       # each distinct value is its own class, lowest class first
        cuts = 0.5 * (u[:-1] + u[1:]) if u.size > 1 else np.array([])
        return np.concatenate([cuts, np.full(k - 1 - cuts.size, np.inf)])
    x = (u - u[0]) / (u[-1] - u[0])      # rescaled to [0, 1] against cancellation in the prefix sums
    w = w.astype(float)
    p0 = np.concatenate([[0.0], np.cumsum(w)])
    p1 = np.concatenate([[0.0], np.cumsum(w * x)])
    p2 = np.concatenate([[0.0], np.cumsum(w * x * x)])
    m = u.size

    def sse(i: np.ndarray, j: int) -> np.ndarray:          # within-class SSE of u[i:j], vectorised in i
        s1 = p1[j] - p1[i]
        return np.clip((p2[j] - p2[i]) - s1 * s1 / (p0[j] - p0[i]), 0.0, None)

    cost = np.full((k, m + 1), np.inf)
    back = np.zeros((k, m + 1), dtype=int)
    cost[0, 1:] = [sse(np.array([0]), j)[0] for j in range(1, m + 1)]
    for c in range(1, k):
        for j in range(c + 1, m + 1):
            i = np.arange(c, j)                              # class c covers u[i:j]
            tot = cost[c - 1, i] + sse(i, j)
            best = int(np.argmax(tot <= tot.min() + 1e-12 * max(tot.min(), 1.0)))   # ties -> earliest cut
            cost[c, j] = tot[best]
            back[c, j] = i[best]
    cuts, j = [], m
    for c in range(k - 1, 0, -1):
        i = back[c, j]
        cuts.append(0.5 * (u[i - 1] + u[i]))
        j = i
    return np.array(cuts[::-1])


def equiprobable_breaks(values: np.ndarray, k: int) -> np.ndarray:
    """Ablation rule: SAX's own principle (every symbol equally likely), empirical quantiles."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.full(k - 1, np.inf)
    return np.quantile(v, np.linspace(0.0, 1.0, k + 1)[1:-1])


@dataclass(frozen=True)
class Quantiles:
    """The three quantile levels of the 'mixed' policy (chosen on TRAINING data, see fit_quantiles.py):
    flat  — quantile of |b| INSIDE THE WINDOW below which a segment is stable;
    rapid — quantile of |b| over the CHANNEL's history (this table) at or above which a moving segment is rapid;
    fluct — quantile of sigma over the channel's history at or above which a segment is fluctuating."""
    flat: float
    rapid: float
    fluct: float


FROZEN_QUANTILES = Quantiles(0.50, 0.95, 0.90)     # selected on training data, see module docstring


@dataclass(frozen=True)
class Thresholds:
    flat: float      # |b| below this            -> Trend S
    rapid: float     # |b| at or above this      -> Rate 2 (between flat and rapid -> Rate 1)
    fluct: float     # sigma at or above this    -> Stability ~


def ground(stats: pd.DataFrame, rule: str = "jenks", tol: float = 0.0) -> Thresholds:
    """Thresholds of ONE channel for ONE table, from that channel's own segment history.
    `tol` is a floating-point tolerance (1e-9 of the channel's value range), not a semantic
    threshold: a fitted change below it is rounding noise of a constant segment."""
    mag, sig = np.abs(stats["b"].to_numpy()), stats["sigma"].to_numpy()
    if rule == "jenks-log":                      # natural breaks of the ORDERS OF MAGNITUDE
        def cut(v, k):
            pos = v[v > tol]                     # below the floating-point floor = exactly flat
            return np.exp(natural_breaks(np.log(pos), k)) if pos.size else np.full(k - 1, np.inf)
        flat, rapid = cut(mag, 3)
        (fluct,) = cut(sig, 2)
        return Thresholds(float(flat), float(rapid), float(fluct))
    fn = natural_breaks if rule == "jenks" else equiprobable_breaks
    flat, rapid = fn(mag, 3)
    (fluct,) = fn(sig, 2)
    return Thresholds(float(flat), float(rapid), float(fluct))


@dataclass(frozen=True)
class Gates:
    move: float      # channel-history quantile of |b| below which nothing counts as moving
    scatter: float   # channel-history quantile of sigma below which nothing counts as fluctuating


LOCAL_GATES = Gates(0.60, 0.80)   # selected on TRAIN + DEV, see the module docstring
RAPID_Q = 0.5                     # quantile of |b| among the moving segments at or above which a segment is rapid


def local_tokens(part: pd.DataFrame, move_gate: float, scatter_gate: float) -> np.ndarray:
    """Trend / Rate / Stability characters of one window under the deployed 'local' policy."""
    b = part["b"].to_numpy(); mag = np.abs(b); sig = part["sigma"].to_numpy()
    flat = max(float(np.median(mag)) if mag.size else np.inf, move_gate)
    moving = (mag >= flat) & (mag > 0)
    mv = mag[moving]
    rapid = float(np.quantile(mv, RAPID_Q)) if mv.size else np.inf   # RAPID_Q = 0.5 is the median (deployed); RQ7 sweeps it
    pos = sig[sig > 0]                                     # a straight segment has no scatter to compare with
    fl = max(float(np.median(pos)) if pos.size else np.inf, scatter_gate)
    trend = np.where(~moving, "S", np.where(b > 0, "R", "F"))
    rate = np.where(~moving, "0", np.where(mag >= rapid, "2", "1"))
    stab = np.where((sig >= fl) & (sig > 0), "~", "=")
    return np.char.add(np.char.add(trend.astype(str), rate.astype(str)), stab.astype(str))


# --------------------------------------------------------------------------- encoders
def _sax_letters(window_means: np.ndarray) -> str:
    sd = float(np.std(window_means))
    if sd < 1e-9:
        return ALPHABET[len(ALPHABET) // 2] * len(window_means)
    z = (window_means - float(np.mean(window_means))) / sd
    return "".join(ALPHABET[i] for i in np.searchsorted(_BREAKPOINTS, z))


def moso_tokens(stats: pd.DataFrame, th: Thresholds) -> pd.Series:
    """The Trend / Rate / Stability characters of every segment (Amplitude is added per window)."""
    b = stats["b"].to_numpy()
    mag = np.abs(b)
    trend = np.where((mag < th.flat) | (mag == 0.0), "S", np.where(b > 0, "R", "F"))   # a zero slope is stable by definition
    rate = np.where(trend == "S", "0", np.where(mag >= th.rapid, "2", "1"))
    stab = np.where(stats["sigma"].to_numpy() >= th.fluct, "~", "=")
    return pd.Series(np.char.add(np.char.add(trend.astype(str), rate.astype(str)), stab.astype(str)), index=stats.index)


def encode_table(series: pd.Series, table: str, rule: str = "jenks", k: int = 0,
                 scope: str = "window", quantiles: "Quantiles | None" = None) -> pd.DataFrame:
    """MOSO strings of one channel for one feature table.

    scope = "channel": thresholds from the channel's whole history in this table;
    scope = "window" : thresholds from the window's own segments (as SAX does for Amplitude);
    scope = "mixed"  : flat = window quantile of |b|, rapid / fluct = channel-history quantiles
                       (levels in `quantiles`, fitted on training data by fit_quantiles.py).
    Returns a frame indexed by window start with columns `sax` (the plain SAX string, for the
    identity check against the pipeline) and `moso` (4 characters per segment)."""
    win_freq, seg_freq = TABLES[table]
    stats = segment_stats(series, seg_freq, win_freq, k)
    # the pipeline interpolates empty segments before SAX; mirror it so Amplitude == SAX
    full = pd.date_range(stats.index.min(), stats.index.max(), freq=seg_freq)
    stats = stats.reindex(full)
    stats["mean"] = stats["mean"].interpolate(method="linear", limit_direction="both").fillna(0.0)
    stats[["n", "b", "sigma"]] = stats[["n", "b", "sigma"]].fillna(0.0)
    vals = series.dropna().to_numpy(dtype=float)
    tol = 1e-9 * float(vals.max() - vals.min()) if vals.size else 0.0
    th_channel = ground(stats[stats["n"] > 0], rule, tol) if scope == "channel" else None
    if scope == "mixed":                           # rapid / fluct from the channel history, flat per window
        hist = stats[stats["n"] > 0]
        q = quantiles or Quantiles(0.2, 0.9, 0.85)
        mag_h, sig_h = np.abs(hist["b"].to_numpy()), hist["sigma"].to_numpy()
        rapid_h = float(np.quantile(mag_h[mag_h > 0], q.rapid)) if (mag_h > 0).any() else np.inf
        fluct_h = float(np.quantile(sig_h[sig_h > 0], q.fluct)) if (sig_h > 0).any() else np.inf
    win = stats.index.to_period(_PERIOD[win_freq]).to_timestamp()
    rows = []
    for w, idx in pd.Series(np.arange(len(stats)), index=win).groupby(level=0):
        part = stats.iloc[idx.to_numpy()]
        if scope == "mixed":
            mag_w = np.abs(part["b"].to_numpy())
            flat_w = float(np.quantile(mag_w, q.flat)) if mag_w.size else np.inf
            th = Thresholds(flat_w, max(rapid_h, flat_w), fluct_h)
        else:
            th = th_channel or ground(part[part["n"] > 0], rule, tol)
        letters = _sax_letters(part["mean"].to_numpy())
        toks = moso_tokens(part, th).to_numpy()
        rows.append((w, letters, "".join(l + t for l, t in zip(letters, toks))))
    out = pd.DataFrame(rows, columns=["window_start", "sax", "moso"]).set_index("window_start")
    out.attrs["thresholds"] = th_channel
    return out


# =========================================================================== baselines
# All baselines share MOSO's windows, segments and Amplitude letters, so the four methods differ
# only in what they add to the plain SAX letter. Definitions follow the original papers
# (checked against the PDFs, 2026-09-22); the one adaptation, common to all methods including
# SAX as deployed in Sonar-TS, is that z-normalisation is done inside each index window.
#
#   SAX     (Lin et al. 2003/2007)      1 char / segment : mean letter a-e
#   1d-SAX  (Malinowski et al. 2013)    2 chars / segment: mean letter a-e + slope letter A-D.
#           Least-squares slope of the z-normalised segment, quantised by the equiprobable
#           breakpoints of N(0, sigma_L^2), sigma_L^2 = 0.03 / L, L = points per segment (their
#           Sec. 3; N_s = 4 as in their experiments). For equally spaced samples their "value of
#           the regression line at the segment centre" equals the segment mean.
#   TFSAX   (Yu et al. 2019)            2 chars / segment: mean letter a-e + trend letter A-E.
#           tan(theta) = (x_end - x_start) / K on the z-normalised segment, K = number of
#           turning points inside the segment (K = 1 if monotone); theta is quantised by their
#           static angle table for alphabet 5: -30, -5, 5, 30 degrees. (The paper writes the
#           denominator once as N and once as "the trend shape factor K"; we follow the text,
#           because with N the trend shape factor the paper introduces would never be used.)
_SLOPE_LETTERS_1D = "ABCD"
_ANGLE_CUTS = np.tan(np.deg2rad([-30.0, -5.0, 5.0, 30.0]))
_TREND_LETTERS_TF = "ABCDE"


def segment_shape(series: pd.Series, seg_freq: str) -> pd.DataFrame:
    """First value, last value and number of turning points K of every segment."""
    s = series.dropna().sort_index()
    seg = s.index.to_period("M").to_timestamp() if seg_freq == "MS" else s.index.floor(seg_freq)
    y = s.to_numpy(dtype=float)
    d = np.diff(y)
    turn = np.zeros(len(y), dtype=float)
    if len(y) > 2:
        prod = d[:-1] * d[1:]
        inner = (prod < 0) | ((prod == 0) & (d[:-1] != d[1:]))
        same = (seg[1:-1] == seg[:-2]) & (seg[1:-1] == seg[2:])     # both neighbours in the same segment
        turn[1:-1] = (inner & np.asarray(same)).astype(float)
    g = pd.DataFrame({"seg": seg, "y": y, "turn": turn}).groupby("seg", sort=True)
    out = g["y"].agg(["first", "last"])
    out["K"] = np.maximum(g["turn"].sum(), 1.0)
    return out


def encode_all(series: pd.Series, table: str, policy: str = "local",
               quantiles: "Quantiles | None" = None) -> pd.DataFrame:
    """The four symbolic strings of one channel for one feature table, indexed by window start:
    columns sax, onedsax, tfsax, moso, plus n_seg (number of segments = tokens in the window).
    policy = "local" (deployed, see the module docstring), "mixed" (ablation; levels =
    FROZEN_QUANTILES unless `quantiles` is given) or "jenks" (ablation: natural breaks inside
    the window)."""
    win_freq, seg_freq = TABLES[table]
    stats = segment_stats(series, seg_freq, win_freq, 0)
    q = quantiles or FROZEN_QUANTILES
    hist = stats[stats["n"] > 0]
    mag_h, sig_h = np.abs(hist["b"].to_numpy()), hist["sigma"].to_numpy()
    qh = lambda a, lvl: float(np.quantile(a[a > 0], lvl)) if (a > 0).any() else np.inf
    if policy == "mixed":
        rapid_h, fluct_h = qh(mag_h, q.rapid), qh(sig_h, q.fluct)
    if policy == "local":
        move_gate, scatter_gate = qh(mag_h, LOCAL_GATES.move), qh(sig_h, LOCAL_GATES.scatter)
    shape = segment_shape(series, seg_freq)
    full = pd.date_range(stats.index.min(), stats.index.max(), freq=seg_freq)
    stats, shape = stats.reindex(full), shape.reindex(full)
    stats["mean"] = stats["mean"].interpolate(method="linear", limit_direction="both").fillna(0.0)
    stats[["n", "b", "sigma"]] = stats[["n", "b", "sigma"]].fillna(0.0)
    shape["K"] = shape["K"].fillna(1.0)
    shape[["first", "last"]] = shape[["first", "last"]].fillna(0.0)
    win = stats.index.to_period(_PERIOD[win_freq]).to_timestamp()
    rows = []
    for w, idx in pd.Series(np.arange(len(stats)), index=win).groupby(level=0):
        pos = idx.to_numpy()
        part, shp = stats.iloc[pos], shape.iloc[pos]
        means = part["mean"].to_numpy()
        sd = float(np.std(means))
        letters = _sax_letters(means)
        # --- MOSO
        if policy == "local":                    # deployed: degree relative to the window, channel gate
            toks = local_tokens(part, move_gate, scatter_gate)
        else:
            if policy == "mixed":                # ablation: rapid / fluct vs. channel history
                mag_w = np.abs(part["b"].to_numpy())
                flat_w = float(np.quantile(mag_w, q.flat)) if mag_w.size else np.inf
                th = Thresholds(flat_w, max(rapid_h, flat_w), fluct_h)
            else:                                # ablation: natural breaks inside the window
                th = ground(part[part["n"] > 0], "jenks")
            toks = moso_tokens(part, th).to_numpy()
        moso = "".join(l + t for l, t in zip(letters, toks))
        # --- 1d-SAX: slope per sample step on the window-z-normalised scale
        L = np.maximum(part["n"].to_numpy(), 1.0)
        slope_z = (part["b"].to_numpy() / L / sd) if sd >= 1e-9 else np.zeros(len(part))
        cuts = norm.ppf([0.25, 0.5, 0.75])[None, :] * np.sqrt(0.03 / L)[:, None]
        s_idx = (slope_z[:, None] > cuts).sum(axis=1)
        oned = "".join(l + _SLOPE_LETTERS_1D[i] for l, i in zip(letters, s_idx))
        # --- TFSAX: end-to-end change divided by the number of turning points
        td = ((shp["last"] - shp["first"]).to_numpy() / sd) if sd >= 1e-9 else np.zeros(len(part))
        t_idx = np.searchsorted(_ANGLE_CUTS, td / shp["K"].to_numpy())
        tf = "".join(l + _TREND_LETTERS_TF[i] for l, i in zip(letters, t_idx))
        rows.append((w, len(pos), letters, oned, tf, moso))
    return pd.DataFrame(rows, columns=["window_start", "n_seg", "sax", "onedsax", "tfsax", "moso"]).set_index("window_start")
