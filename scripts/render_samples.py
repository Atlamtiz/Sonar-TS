"""Render one sample PNG per subtask from the current predict.json (or GT)."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sonar_ts._paths import PROJECT_ROOT, TASKS_JSON
from sonar_ts.postprocess.visualize import save_figure


SUBTASK_ORDER = [
    "Global Aggregation",
    "Temporal Localization",
    "Interval Discovery",
    "Sliding Window",
    "Shape Identification",
    "Periodicity Detection",
    "Subsequence Matching",
    "Composite Trend",
    "Contextual Anomaly",
    "Causal Anomaly",
    "Insight Synthesis",
]


def _load_predictions() -> dict[int, str]:
    """Map row_idx -> prediction string from output/predict.json."""
    out: dict[int, str] = {}
    for candidate in (PROJECT_ROOT / "output" / "predict.json",
                      PROJECT_ROOT / "predict.json"):
        if candidate.exists():
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                for entry in raw:
                    try:
                        idx = int(entry["id"])
                        out[idx] = entry.get("prediction", "")
                    except (KeyError, TypeError, ValueError):
                        continue
            except json.JSONDecodeError:
                pass
            break
    return out


def _parse_to_result(prediction: str, eval_metric: str):
    """Coerce the predict.json string back into a Python value usable by
    the renderers (which expect raw _result, not the formatted string)."""
    if prediction is None or prediction == "":
        return None
    pred = prediction.strip()
    if eval_metric == "rel_acc":
        # extract a number
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
        # try to parse into the structured dict the renderer expects.
        # If predict.json carries the LLM's text, we re-parse it back.
        segments = []
        seg_re = re.compile(
            r"from\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+to\s+"
            r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*,?\s*the\s+trend\s+showed\s+a\s+(\w+)\s+(\w+)",
            re.IGNORECASE,
        )
        for m in seg_re.finditer(pred.replace("\n", " ")):
            segments.append({"start": m.group(1), "end": m.group(2),
                             "adj":   m.group(3).lower(),
                             "kind":  m.group(4).lower()})
        outliers = []
        out_re = re.compile(r"detected\s+at\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
                            re.IGNORECASE)
        for m in out_re.finditer(pred.replace("\n", " ")):
            outliers.append({"timestamp": m.group(1), "value": 0.0})
        return {"trend_segments": segments, "outliers": outliers}
    return pred


def main() -> None:
    tasks = json.load(open(TASKS_JSON, encoding="utf-8"))
    preds = _load_predictions()
    if preds:
        print(f"Loaded {len(preds)} predictions from output/predict.json")
    else:
        print("No predict.json found; falling back to ground-truth answers.")

    # Pick one task per subtask. Prefer tasks that actually have a
    # prediction; deterministic seed for reproducible sampling.
    random.seed(2026)
    samples_per_subtask: dict[str, dict] = {}
    indexed = list(enumerate(tasks))
    random.shuffle(indexed)
    for idx, t in indexed:
        st = t.get("subtask", "")
        if st in samples_per_subtask:
            continue
        # prefer non-empty predictions
        if preds and not preds.get(idx, "").strip():
            continue
        samples_per_subtask[st] = {"idx": idx, "task": t}
    # Fill remaining subtasks if any didn't get a sample with prediction.
    for idx, t in enumerate(tasks):
        st = t.get("subtask", "")
        if st not in samples_per_subtask:
            samples_per_subtask[st] = {"idx": idx, "task": t}

    out_dir = PROJECT_ROOT / "output" / "figures" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering one sample per subtask to: {out_dir}\n")
    print(f"  {'Subtask':<24s}  {'task_id':<45s}  {'pred?':>5s}  {'png_kb':>7s}")
    for st in SUBTASK_ORDER:
        entry = samples_per_subtask.get(st)
        if entry is None:
            print(f"  {st:<24s}  (no task for this subtask)")
            continue
        task = entry["task"]
        idx = entry["idx"]
        prediction = preds.get(idx, task.get("answer", ""))
        result = _parse_to_result(prediction, task["eval_metric"])
        out_png = out_dir / f"{task['id']}.png"
        save_figure(task, result, out_png)
        pred_flag = "yes" if preds.get(idx, "").strip() else "GT  "
        size_kb = out_png.stat().st_size // 1024 if out_png.exists() else 0
        print(f"  {st:<24s}  {task['id']:<45s}  {pred_flag:>5s}  {size_kb:>7d}")

    print(f"\nDone. Inspect PNGs under: {out_dir}")


if __name__ == "__main__":
    main()
