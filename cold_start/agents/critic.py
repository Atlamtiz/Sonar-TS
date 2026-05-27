"""Critic agent: review a draft skill and return structured feedback."""

from __future__ import annotations

from typing import List

from sonar_ts.llm import DeepSeekClient
from .reflector import TaskExample


_SYSTEM_PROMPT = """You are reviewing a methodology skill markdown that
will be injected into an LLM code generator for the Sonar-TS framework.

# FRAMEWORK FACTS — verify the draft against these

* Runtime globals: `conn` (sqlite3 to per-task db), `pd`, `np`, `sqlite3`,
  `json`, `math`, `re`, `datetime`, `timedelta`. The skill must NOT
  assume any other inputs (e.g. NEVER mention `df_pred` / `df_true` —
  the framework does not pass them).
* Data access: `raw_data` table is WIDE — columns are
  `(timestamp, <channel_1>, ...)`. SQL reads need double-quoted channel
  names: `SELECT timestamp, "<channel>" FROM raw_data ...`.
* The script ends with `_result = ...`; the framework appends the JSON
  print. The expected `_result` shape per eval_metric:
    rel_acc → number               hit → "YYYY-MM-DD HH:MM:SS"
    iou     → [start_ts, end_ts]   set_f1 → ["YYYY-MM-DD", ...]
    report  → {"trend_segments": [...], "outliers": [...]}

# REVIEW CRITERIA — judge on these axes

1. **Framework fit** — does the draft respect runtime globals + table
   layout + `_result` schema above? Flag any hallucinated input names.
2. **Correctness** — would the algorithm compute the right answer for
   the training examples shown?
3. **Specificity** — are operations / thresholds / window sizes concrete,
   not hand-wavy?
4. **Edge cases** — does it handle empty / short-history / single-channel
   / boundary cases that broke earlier versions?

# OUTPUT FORMAT

A structured bullet list. Each bullet is either:
- "OK: <axis>"  ... when that axis is fine, OR
- "FIX: <axis>: <concrete change>"  ... when something needs changing.

Be SPECIFIC about fixes — name the line, the variable, or the missing
step. No generic "consider tightening" advice. 4-8 bullets total.
"""


def _render_examples(items: List[TaskExample]) -> str:
    if not items:
        return "(none provided)"
    out = []
    for i, ex in enumerate(items, 1):
        out.append(
            f"[{i}] metric={ex.eval_metric} score={ex.score:.3f}\n"
            f"  Q:  {ex.question}\n"
            f"  GT: {ex.ground_truth}\n"
            f"  P:  {ex.prediction or '(empty)'}"
        )
    return "\n".join(out)


class CriticAgent:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def review(self, subtask: str, eval_metric: str, draft_skill: str,
               examples: List[TaskExample]) -> str:
        """Return the LLM's structured critique."""
        user = (
            f"Sub-task: **{subtask}**\n"
            f"Eval metric: `{eval_metric}`\n\n"
            f"### Skill draft under review:\n{draft_skill}\n\n"
            f"### Training examples for context (mix of pass/fail):\n"
            f"{_render_examples(examples)}\n\n"
            "Review the draft against the criteria. Output the bullet list."
        )
        return self.client.chat([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]).strip()
