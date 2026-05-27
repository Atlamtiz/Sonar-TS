"""Task planner."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ._paths import PROJECT_ROOT
from .llm import DeepSeekClient

_PROMPT_PATH = PROJECT_ROOT / "sonar_ts" / "prompts" / "task_planner.txt"

_SKILLS_RE = re.compile(
    r"SKILLS_USED\s*:\s*\[?\s*([^\]\n]*?)\s*\]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class PlanResult:
    plan_text: str
    skill_ids: List[str]


def _parse_skills_line(raw: str) -> Tuple[str, List[str]]:
    m = _SKILLS_RE.search(raw)
    if m is None:
        return raw.strip(), []
    ids_blob = m.group(1)
    plan = (raw[:m.start()] + raw[m.end():]).strip()
    ids = []
    for tok in ids_blob.split(","):
        s = tok.strip().strip("'\"`")
        if s:
            ids.append(s)
    return plan, ids


def plan(question: str, schema: str, manifest: str,
         client: DeepSeekClient) -> PlanResult:
    prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
        manifest=manifest or "(no skills available)",
        schema=schema,
        question=question,
    )
    raw = client.chat([{"role": "user", "content": prompt}])
    plan_text, skill_ids = _parse_skills_line(raw)
    return PlanResult(plan_text=plan_text, skill_ids=skill_ids)
