"""Reflector agent: extract error patterns from failures and successes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sonar_ts.llm import DeepSeekClient


_SYSTEM_PROMPT = """You are an expert ML engineer reviewing a time-series QA
pipeline's output. The pipeline is given (question, schema, methodology) and
emits Python code that runs against a SQLite database to produce an answer.

You will be shown FAILED examples (low score) and a couple of SUCCESSFUL
examples (high score) for ONE sub-task type. Your job is to identify the
*recurring* patterns of error — what the code is consistently missing or
mishandling — and contrast them with what the successes get right.

Output STRICT format: a bulleted list (one pattern per bullet).
- Each bullet starts with "- " and ends with a period.
- Bullets must be CONCRETE and ACTIONABLE — name the data shape, the
  algorithm step, or the boundary condition involved. Avoid vague advice
  like "improve handling" or "be more careful".
- 3-7 bullets total. No preamble, no closing remarks, just the list.
"""


@dataclass
class TaskExample:
    question: str
    ground_truth: str
    prediction: str
    score: float
    eval_metric: str


def _render_examples(label: str, items: List[TaskExample]) -> str:
    if not items:
        return f"### {label} examples\n(none)\n"
    lines = [f"### {label} examples ({len(items)})"]
    for i, ex in enumerate(items, 1):
        lines.append(
            f"\n[{i}] score={ex.score:.3f}  metric={ex.eval_metric}\n"
            f"Q:  {ex.question}\n"
            f"GT: {ex.ground_truth}\n"
            f"P:  {ex.prediction or '(empty)'}"
        )
    return "\n".join(lines)


class ReflectorAgent:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def reflect(self, subtask: str, eval_metric: str,
                failures: List[TaskExample],
                successes: List[TaskExample]) -> str:
        user = (
            f"Sub-task: **{subtask}**\n"
            f"Eval metric: `{eval_metric}`\n\n"
            f"{_render_examples('FAILED', failures)}\n\n"
            f"{_render_examples('SUCCESSFUL', successes)}\n\n"
            "Identify the recurring patterns of error in the failures, "
            "contrasted with the successes. Output the bulleted list only."
        )
        return self.client.chat([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]).strip()
