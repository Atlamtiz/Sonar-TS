"""Format predictions for the evaluator and parse them back for visualization."""

from __future__ import annotations

import re
from typing import Any

from .formatters import format_prediction

__all__ = ["format_prediction", "parse_prediction_for_viz"]


def parse_prediction_for_viz(prediction: str, eval_metric: str) -> Any:
    if prediction is None:
        return None
    pred = str(prediction).strip()
    if not pred:
        return None

    if eval_metric == "rel_acc":
        m = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", pred)
        try:
            return float(m[-1]) if m else None
        except ValueError:
            return None

    if eval_metric == "hit":
        m = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", pred)
        return m[0] if m else None

    if eval_metric == "iou":
        m = re.findall(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?", pred)
        return [m[0], m[1]] if len(m) >= 2 else None

    if eval_metric == "set_f1":
        return re.findall(r"\d{4}-\d{2}-\d{2}", pred)

    if eval_metric == "report":
        segments = []
        seg_re = re.compile(
            r"from\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+to\s+"
            r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*,?\s*"
            r"the\s+trend\s+showed\s+a\s+(\w+)\s+(\w+)",
            re.IGNORECASE,
        )
        flat = pred.replace("\n", " ")
        for m in seg_re.finditer(flat):
            segments.append({
                "start": m.group(1),
                "end":   m.group(2),
                "adj":   m.group(3).lower(),
                "kind":  m.group(4).lower(),
            })
        outliers = []
        out_re = re.compile(
            r"detected\s+at\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
            re.IGNORECASE,
        )
        for m in out_re.finditer(flat):
            outliers.append({"timestamp": m.group(1), "value": None})
        return {"trend_segments": segments, "outliers": outliers}

    return pred
