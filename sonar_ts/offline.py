"""Offline feature-table construction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm


def generate_breakpoints(alphabet_size: int) -> np.ndarray:
    q_points = np.linspace(0.0, 1.0, alphabet_size + 1)[1:-1]
    return norm.ppf(q_points)


def compute_local_sax(paa_values: np.ndarray, alphabet: List[str],
                      breakpoints: np.ndarray) -> str:
    if len(paa_values) == 0:
        return ""
    mean = float(np.mean(paa_values))
    std = float(np.std(paa_values))
    if std < 1e-9:
        mid = alphabet[len(alphabet) // 2]
        return mid * len(paa_values)
    z = (paa_values - mean) / std
    indices = np.searchsorted(breakpoints, z)
    return "".join(alphabet[i] for i in indices)


def compute_slope(series: pd.Series) -> float:
    y = series.values
    n = len(y)
    if n < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, n)
    y_mean = float(np.mean(y))
    num = float(np.sum((x - 0.5) * (y - y_mean)))
    den = float(np.sum((x - 0.5) ** 2))
    return 0.0 if den == 0 else num / den


def compute_features_for_view(
    raw_long: pd.DataFrame,
    *,
    freq: str,
    resample_rule: str,
    alphabet: List[str],
    breakpoints: np.ndarray,
) -> pd.DataFrame:
    if raw_long.empty:
        return pd.DataFrame()

    df = raw_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    pieces: list[pd.DataFrame] = []
    for channel_id, group in df.groupby("channel_id", sort=False):
        series = group.set_index("timestamp")["value"].sort_index().dropna()
        if series.empty:
            continue

        r = series.resample(freq)
        stats = r.agg(["min", "max", "mean", "std"])
        stats.columns = ["min_val", "max_val", "avg_val", "std_val"]
        stats = stats.dropna()
        if stats.empty:
            continue
        slope_series = r.apply(compute_slope).loc[stats.index]

        paa_full = series.resample(resample_rule).mean()
        paa_full = paa_full.interpolate(method="linear",
                                        limit_direction="both").fillna(0.0)

        def _sax(chunk: pd.Series) -> str | None:
            if len(chunk) == 0:
                return None
            return compute_local_sax(chunk.values, alphabet, breakpoints)

        sax_series = paa_full.resample(freq).apply(_sax)
        sax_series = sax_series.reindex(stats.index).dropna()

        common = stats.index.intersection(sax_series.index)
        if common.empty:
            continue

        feat = pd.DataFrame(index=common)
        feat["channel_id"] = channel_id
        feat["window_start"] = common
        feat["window_end"] = common + pd.tseries.frequencies.to_offset(freq)
        feat["min_val"] = stats.loc[common, "min_val"]
        feat["max_val"] = stats.loc[common, "max_val"]
        feat["avg_val"] = stats.loc[common, "avg_val"]
        feat["std_val"] = stats.loc[common, "std_val"]
        feat["slope"] = slope_series.loc[common]
        feat["sax"] = sax_series.loc[common]
        feat["sax_len"] = feat["sax"].str.len()
        pieces.append(feat.reset_index(drop=True))

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    out["window_start"] = out["window_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["window_end"] = out["window_end"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out[["channel_id", "window_start", "window_end",
                "min_val", "max_val", "avg_val", "std_val",
                "slope", "sax_len", "sax"]]


def compute_all_features(
    raw_long: pd.DataFrame,
    view_config: Dict[str, Dict[str, str]],
    alphabet: List[str],
) -> Dict[str, pd.DataFrame]:
    breakpoints = generate_breakpoints(len(alphabet))
    out: Dict[str, pd.DataFrame] = {}
    for view_name, settings in view_config.items():
        df = compute_features_for_view(
            raw_long,
            freq=settings["freq"],
            resample_rule=settings["resample_rule"],
            alphabet=alphabet,
            breakpoints=breakpoints,
        )
        out[f"{view_name}_feature"] = df
    return out
