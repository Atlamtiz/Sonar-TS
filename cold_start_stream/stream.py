"""Label-blind stream loader.

The streaming cold start must work on an *unlabeled query log*: everything
that betrays how the benchmark was built (`subtask`, `category`, `level`,
`level_name`, `meta`, the text of the task id) is stripped here and never
reaches a decision or a prompt. Items are shuffled once with a fixed seed and
get opaque uids `q00001…`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Anything that looks like a benchmark task id (`L2_T3_Shape_...`).
_TASK_ID_RE = re.compile(r"\bL[1-4]_[A-Za-z0-9_]+")

# A timestamp inside a free-text ground truth (`['2020-01-01 00:00:00', …]`).
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


@dataclass(frozen=True)
class StreamItem:
    """One query-log entry as the procedure is allowed to see it."""

    uid: str
    question: str
    eval_metric: str
    ground_truth: Any
    task: Dict[str, Any] = field(repr=False, default_factory=dict)
    # ^ the raw task dict, used ONLY to call run_task / locate the database.

    @property
    def task_id(self) -> str:
        """Internal plumbing only — never put this in a prompt."""
        return str(self.task.get("id", ""))

    @property
    def channel(self) -> str:
        return str(self.task.get("channel", ""))


def interval_ends(value: Any) -> Tuple[str, str] | None:
    """The two endpoints of an interval ground truth, as written."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]).strip(), str(value[1]).strip()
    found = _TS_RE.findall(str(value))
    if len(found) >= 2:
        return found[0].strip(), found[1].strip()
    return None


def unscorable_reason(task: Dict[str, Any]) -> str | None:
    """Why this item can never be scored — or None if it can.

    Label-free: it looks only at `eval_metric` and the ground truth, both of
    which the procedure is allowed to see anyway.
    """
    gt = task.get("ground_truth", task.get("answer"))
    if gt is None:
        return "empty_ground_truth"
    if isinstance(gt, (str, bytes)):
        if not str(gt).strip() or str(gt).strip() in {"[]", "{}", '""', "''"}:
            return "empty_ground_truth"
    elif isinstance(gt, (list, tuple, set, dict)) and len(gt) == 0:
        return "empty_ground_truth"
    if str(task.get("eval_metric", "")).strip().lower() == "iou":
        ends = interval_ends(gt)
        if ends is not None and ends[0] == ends[1]:
            # A zero-length interval: every prediction scores 0.0, so the item
            # only adds noise to d_fail.
            return "degenerate_iou"
    return None


def load_stream(tasks_json: Path | str, *, seed: int, limit: int = 0,
                stats: Dict[str, Any] | None = None,
                ids: Iterable[str] | None = None) -> List[StreamItem]:
    """Read tasks.json, drop unscorable items, shuffle, assign opaque uids.

    The sanity filter runs BEFORE the shuffle so that uids stay contiguous.
    `stats`, when given, is filled with the counts of what was dropped.
    `ids` (ADDENDUM E1/E2) restricts the stream to these benchmark task ids —
    the label-free TRAIN half of the split; the restriction is applied before
    the sanity filter and the shuffle, so the stream of a given id set and
    seed does not depend on what the other half contains.
    """
    tasks = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    wanted = None if ids is None else {str(i) for i in ids}
    if wanted is not None:
        selected = [t for t in tasks if str(t.get("id")) in wanted]
    else:
        selected = list(tasks)
    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    for t in selected:
        reason = unscorable_reason(t)
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        kept.append(t)
    if stats is not None:
        stats.update({"total": len(tasks), "kept": len(kept),
                      "skipped": sum(dropped.values()),
                      "skipped_by_reason": dropped})
        # Only an id-restricted stream reports the restriction, so the stats of
        # an ordinary run are exactly what they were before E2.
        if wanted is not None:
            found = {str(t.get("id")) for t in selected}
            stats["selected"] = len(selected)
            stats["ids_requested"] = len(wanted)
            stats["ids_not_found"] = sorted(wanted - found)

    order = list(range(len(kept)))
    random.Random(seed).shuffle(order)
    if limit:
        order = order[:limit]
    items: List[StreamItem] = []
    for n, idx in enumerate(order, start=1):
        t = kept[idx]
        items.append(StreamItem(
            uid=f"q{n:05d}",
            question=t["question"],
            eval_metric=t["eval_metric"],
            ground_truth=t.get("ground_truth", t.get("answer")),
            task=t,
        ))
    return items


def channel_literals(items: List[StreamItem]) -> set[str]:
    """Channel identifiers occurring in the stream (used by the linter)."""
    out: set[str] = set()
    for it in items:
        ch = it.channel.strip()
        if ch:
            out.add(ch)
    return out


def scrub(text: Any) -> str:
    """Remove benchmark task ids from text that is about to enter a prompt."""
    s = "" if text is None else str(text)
    return _TASK_ID_RE.sub("<case>", s)
