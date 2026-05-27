"""NLQTSBench scoring and aggregation."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import pandas as pd

_EPS = 1e-9

_NUM_RE = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")
_TS_FULL_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def _clean(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    for ch in "'\"[]()":
        s = s.replace(ch, "")
    return s.strip()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    matches = _NUM_RE.findall(str(value))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value
    s = _clean(value).replace("T", " ").replace("Z", "")
    if not s:
        return None
    m = _TS_FULL_RE.search(str(value))
    candidate = m.group(0).replace("T", " ").replace("Z", "") if m else s
    for fmt in _TS_FORMATS:
        try:
            ts = pd.Timestamp(pd.to_datetime(candidate, format=fmt))
            return None if pd.isna(ts) else ts
        except (ValueError, TypeError):
            continue
    try:
        ts = pd.Timestamp(candidate)
        return None if pd.isna(ts) else ts
    except (ValueError, TypeError):
        return None


def parse_interval(value: Any) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        a = parse_timestamp(value[0])
        b = parse_timestamp(value[1])
        if a is not None and b is not None:
            return (a, b) if a <= b else (b, a)
    matches = _TS_FULL_RE.findall(str(value))
    if len(matches) >= 2:
        a = parse_timestamp(matches[0])
        b = parse_timestamp(matches[1])
        if a is not None and b is not None:
            return (a, b) if a <= b else (b, a)
    return None


def parse_date_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        flat: List[Any] = []
        for v in value:
            if isinstance(v, (list, tuple, set)):
                flat.extend(v)
            else:
                flat.append(v)
        return {str(v)[:10] for v in flat if _DATE_RE.match(str(v)[:10])}
    return set(_DATE_RE.findall(str(value)))


def score_relative_accuracy(pred: Any, gt: Any) -> float:
    p, g = parse_float(pred), parse_float(gt)
    if p is None or g is None:
        return 0.0
    return max(0.0, 1.0 - abs(p - g) / (abs(g) + _EPS))


def score_hit(pred: Any, gt: Any, *, delta: timedelta = timedelta(0)) -> float:
    p, g = parse_timestamp(pred), parse_timestamp(gt)
    if p is None or g is None:
        return 0.0
    return 1.0 if abs(p - g) <= delta else 0.0


def score_iou(pred: Any, gt: Any) -> float:
    p = parse_interval(pred)
    g = parse_interval(gt)
    if p is None or g is None:
        return 0.0
    ps, pe = p
    gs, ge = g
    inter = max(timedelta(0), min(pe, ge) - max(ps, gs)).total_seconds()
    union = ((pe - ps) + (ge - gs)).total_seconds() - inter
    if union > 0:
        return inter / union
    return 1.0 if (ps == gs and pe == ge) else 0.0


def score_set_f1(pred: Any, gt: Any) -> float:
    p = parse_date_set(pred)
    g = parse_date_set(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2.0 * precision * recall / (precision + recall)


REPORT_WEIGHTS = {"trend": 0.40, "interval": 0.30, "adjective": 0.20, "outlier": 0.10}
OUTLIER_TOLERANCE = timedelta(hours=4)

_SEG_RE = re.compile(
    r"from\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+to\s+"
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*,?\s*"
    r"the\s+trend\s+showed\s+a\s+([a-zA-Z]+)\s+([a-zA-Z]+)",
    re.IGNORECASE,
)
_OUT_RE = re.compile(
    r"detected\s+at\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)


def _parse_report_text(text: str) -> Tuple[List[dict], List[pd.Timestamp]]:
    text_one_line = (text or "").replace("\n", " ")
    segs: List[dict] = []
    for m in _SEG_RE.finditer(text_one_line):
        s, e, adj, kind = m.groups()
        try:
            segs.append({
                "start": pd.Timestamp(s.replace("T", " ")),
                "end":   pd.Timestamp(e.replace("T", " ")),
                "adj":   adj.lower(),
                "kind":  kind.lower(),
            })
        except (ValueError, TypeError):
            continue
    outs: List[pd.Timestamp] = []
    for m in _OUT_RE.finditer(text_one_line):
        try:
            outs.append(pd.Timestamp(m.group(1).replace("T", " ")))
        except (ValueError, TypeError):
            continue
    return segs, outs


def _parse_report_gt(gt: Any) -> Tuple[List[dict], List[pd.Timestamp]]:
    if isinstance(gt, dict):
        joined = " ".join(gt.get("trend_segments", []))
        segs, _ = _parse_report_text(joined)
        outs: List[pd.Timestamp] = []
        sig = gt.get("significant_anomaly") or {}
        ts = sig.get("timestamp")
        if ts is not None:
            try:
                outs.append(pd.Timestamp(ts))
            except (ValueError, TypeError):
                pass
        return segs, outs
    return _parse_report_text(str(gt))


def _iou_seconds(a_s, a_e, b_s, b_e) -> float:
    start = max(a_s, b_s)
    end = min(a_e, b_e)
    inter = max(timedelta(0), end - start).total_seconds()
    union = ((a_e - a_s) + (b_e - b_s)).total_seconds() - inter
    return inter / union if union > 0 else 0.0


def score_report(pred: Any, gt: Any) -> Dict[str, float]:
    pred_segs, pred_outs = _parse_report_text(str(pred) if pred is not None else "")
    gt_segs, gt_outs = _parse_report_gt(gt)

    pred_kinds = [s["kind"] for s in pred_segs]
    gt_kinds = [s["kind"] for s in gt_segs]
    matcher = SequenceMatcher(None, gt_kinds, pred_kinds)
    matched: List[Tuple[int, int]] = []
    hits = 0
    for blk in matcher.get_matching_blocks():
        for i in range(blk.size):
            matched.append((blk.a + i, blk.b + i))
        hits += blk.size
    denom = len(gt_kinds) + len(pred_kinds)
    s_trend = 1.0 if denom == 0 else 2.0 * hits / denom

    if hits > 0:
        sum_iou, sum_adj = 0.0, 0.0
        for g_i, p_i in matched:
            g, p = gt_segs[g_i], pred_segs[p_i]
            sum_iou += _iou_seconds(g["start"], g["end"], p["start"], p["end"])
            sum_adj += 1.0 if g["adj"] == p["adj"] else 0.0
        s_interval = sum_iou / hits
        s_adj = sum_adj / hits
    else:
        s_interval = s_adj = 0.0

    if not gt_outs and not pred_outs:
        s_out = 1.0
    elif not gt_outs or not pred_outs:
        s_out = 0.0
    else:
        tp = 0
        used: set[int] = set()
        for p in pred_outs:
            for i, g in enumerate(gt_outs):
                if i not in used and abs(p - g) <= OUTLIER_TOLERANCE:
                    tp += 1
                    used.add(i)
                    break
        prec = tp / len(pred_outs)
        rec = tp / len(gt_outs)
        s_out = 0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec)

    overall = (s_trend * REPORT_WEIGHTS["trend"]
               + s_interval * REPORT_WEIGHTS["interval"]
               + s_adj * REPORT_WEIGHTS["adjective"]
               + s_out * REPORT_WEIGHTS["outlier"])
    return {
        "overall": overall,
        "trend": s_trend, "interval": s_interval,
        "adjective": s_adj, "outlier": s_out,
    }


def score_one(eval_metric: str, pred: Any, gt: Any
              ) -> Tuple[float, Dict[str, float] | None]:
    if eval_metric == "rel_acc":
        return score_relative_accuracy(pred, gt), None
    if eval_metric == "hit":
        return score_hit(pred, gt), None
    if eval_metric == "iou":
        return score_iou(pred, gt), None
    if eval_metric == "set_f1":
        return score_set_f1(pred, gt), None
    if eval_metric == "report":
        br = score_report(pred, gt)
        return br["overall"], br
    raise ValueError(f"unknown eval_metric: {eval_metric}")


CATEGORY_ORDER: List[Tuple[str, str, str]] = [
    ("Atomic Retrieval",      "L1", "AR"),
    ("Sliding Window",        "L1", "SW"),
    ("Shape Identification",  "L2", "SI"),
    ("Periodicity Detection", "L2", "PD"),
    ("Subsequence Matching",  "L2", "SM"),
    ("Composite Trend",       "L3", "CT"),
    ("Contextual Anomaly",    "L3", "CxA"),
    ("Causal Anomaly",        "L3", "CsA"),
    ("Insight Synthesis",     "L4", "IS"),
]

SUBTASK_METRIC = {
    "Global Aggregation":    "rel_acc",
    "Temporal Localization": "hit",
    "Interval Discovery":    "iou",
    "Sliding Window":        "iou",
    "Shape Identification":  "iou",
    "Periodicity Detection": "rel_acc",
    "Subsequence Matching":  "iou",
    "Composite Trend":       "set_f1",
    "Contextual Anomaly":    "iou",
    "Causal Anomaly":        "iou",
    "Insight Synthesis":     "report",
}


def aggregate(per_row: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_subtask: dict = defaultdict(list)
    by_category: dict = defaultdict(list)
    by_level: dict = defaultdict(list)
    all_scores: List[float] = []
    for r in per_row:
        s = r["score"]
        by_subtask[r["subtask"]].append(s)
        by_category[r["category"]].append(s)
        by_level[int(r["level"])].append(s)
        all_scores.append(s)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "overall":     {"n": len(all_scores), "avg": mean(all_scores)},
        "by_level":    {k: {"n": len(v), "avg": mean(v)} for k, v in by_level.items()},
        "by_category": {k: {"n": len(v), "avg": mean(v)} for k, v in by_category.items()},
        "by_subtask":  {k: {"n": len(v), "avg": mean(v)} for k, v in by_subtask.items()},
    }


def _print_report(agg: Dict[str, Any]) -> None:
    bar = "=" * 78
    print()
    print(bar)
    print("  NLQTSBench evaluation")
    print(bar)

    print("\n  Per-category scores (paper alignment)")
    print(f"  {'Category':24s} {'Code':6s} {'Level':6s} {'N':>5s}  {'Score':>8s}")
    print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*5}  {'-'*8}")
    for cat, lvl, code in CATEGORY_ORDER:
        info = agg["by_category"].get(cat, {"n": 0, "avg": 0.0})
        print(f"  {cat:24s} {code:6s} {lvl:6s} {info['n']:>5d}  {info['avg']:>8.4f}")

    print("\n  Per-level scores")
    print(f"  {'Level':6s} {'N':>5s}  {'Score':>8s}")
    print(f"  {'-'*6} {'-'*5}  {'-'*8}")
    for lvl in sorted(agg["by_level"]):
        info = agg["by_level"][lvl]
        print(f"  L{lvl:<5d} {info['n']:>5d}  {info['avg']:>8.4f}")

    print()
    print(f"  Overall  N = {agg['overall']['n']:>5d}   "
          f"Score = {agg['overall']['avg']:.4f}")
    print(bar)


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(
        prog="python -m sonar_ts.evaluator",
        description="Score a NLQTSBench submission against ground truth. "
                    "No LLM pipeline involved.",
    )
    p.add_argument("--tasks", required=True,
                   help="Path to tasks.json (the benchmark spec).")
    p.add_argument("--predict", required=True,
                   help="Path to your predict.json submission.")
    p.add_argument("--out-summary", default=None,
                   help="Optional output path for the paper-aligned "
                        "summary JSON (per-category / per-level / overall).")
    p.add_argument("--out-per-task", default=None,
                   help="Optional output path for per-task results "
                        "(prediction + score + breakdown per row).")
    args = p.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    preds_raw = json.loads(Path(args.predict).read_text(encoding="utf-8"))
    predictions: Dict[int, str] = {}
    for entry in preds_raw:
        try:
            predictions[int(entry["id"])] = entry.get("prediction", "")
        except (KeyError, TypeError, ValueError):
            continue

    per_row: List[Dict[str, Any]] = []
    for idx, t in enumerate(tasks):
        pred = predictions.get(idx, "")
        try:
            score, breakdown = score_one(t["eval_metric"], pred,
                                          t.get("ground_truth", t.get("answer")))
        except Exception:  # noqa: BLE001
            score, breakdown = 0.0, None
        per_row.append({
            "id":          idx,
            "task_id":     t["id"],
            "level":       t["level"],
            "category":    t["category"],
            "subtask":     t["subtask"],
            "eval_metric": t["eval_metric"],
            "prediction":  pred,
            "ground_truth": t.get("answer", t.get("ground_truth")),
            "score":       float(score),
            "breakdown":   breakdown,
        })

    agg = aggregate(per_row)
    _print_report(agg)

    print(f"\n  Scored {len(per_row)} tasks; "
          f"missing predictions: {sum(1 for i in range(len(tasks)) if i not in predictions)}")

    if args.out_summary:
        summary = {
            "overall": agg["overall"],
            "by_level": {f"L{k}": v for k, v in sorted(agg["by_level"].items())},
            "by_category": {
                cat: {"code": code, "level": lvl,
                      **agg["by_category"].get(cat, {"n": 0, "avg": 0.0})}
                for cat, lvl, code in CATEGORY_ORDER
            },
            "by_subtask": {
                st: {"metric": SUBTASK_METRIC[st],
                     **agg["by_subtask"].get(st, {"n": 0, "avg": 0.0})}
                for st in SUBTASK_METRIC
            },
        }
        Path(args.out_summary).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"  Wrote summary  → {args.out_summary}")

    if args.out_per_task:
        Path(args.out_per_task).write_text(
            json.dumps(per_row, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"  Wrote per-task → {args.out_per_task}")


if __name__ == "__main__":
    main()
