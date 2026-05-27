"""Code generator."""

from __future__ import annotations

import re
from pathlib import Path

from ._paths import PROJECT_ROOT
from .llm import DeepSeekClient

_PROMPT_DIR = PROJECT_ROOT / "sonar_ts" / "prompts"
_INITIAL = _PROMPT_DIR / "code_generator.txt"
_FIX = _PROMPT_DIR / "code_generator_fix.txt"

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    return (m.group(1) if m else text).strip()


def generate(question: str, schema: str, plan: str, experiences: str,
             eval_metric: str, client: DeepSeekClient) -> str:
    prompt = _INITIAL.read_text(encoding="utf-8").format(
        schema=schema,
        experiences=experiences or "(none)",
        question=question,
        plan=plan,
        eval_metric=eval_metric,
    )
    raw = client.chat([{"role": "user", "content": prompt}])
    return _extract_code(raw)


def regenerate(question: str, schema: str, plan: str, experiences: str,
               eval_metric: str,
               previous_code: str, error_status: str, error_message: str,
               client: DeepSeekClient) -> str:
    prompt = _FIX.read_text(encoding="utf-8").format(
        schema=schema,
        experiences=experiences or "(none)",
        question=question,
        plan=plan,
        eval_metric=eval_metric,
        previous_code=previous_code,
        error_status=error_status,
        error_message=error_message,
    )
    raw = client.chat([{"role": "user", "content": prompt}])
    return _extract_code(raw)
