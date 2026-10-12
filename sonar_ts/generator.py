"""Code generator."""

from __future__ import annotations

import os
import re
from pathlib import Path

from ._paths import PROJECT_ROOT
from .index_guide import load_index_guide, resolve_symbol_column
from .llm import DeepSeekClient
from .usage import llm_tag

_PROMPT_DIR = PROJECT_ROOT / "sonar_ts" / "prompts"
_INITIAL = _PROMPT_DIR / "code_generator.txt"
_FIX = _PROMPT_DIR / "code_generator_fix.txt"
# Ablation "w/o Verification": no Python at all. The model writes ONE SQL query;
# a fixed wrapper (prompts/sql_wrapper.py.txt) runs it and maps the result to
# the answer shape, nothing else. Switched on by SONAR_SQL_ONLY=1.
_SQL_ONLY_ENV = "SONAR_SQL_ONLY"
_INITIAL_SQL = _PROMPT_DIR / "code_generator_sqlonly.txt"
_FIX_SQL = _PROMPT_DIR / "code_generator_fix_sqlonly.txt"
_SQL_WRAPPER_FILE = _PROMPT_DIR / "sql_wrapper.py.txt"
_SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.DOTALL)
_SQL_IN_WRAPPER_RE = re.compile(r'SQL = """(.*?)"""', re.DOTALL)


def sql_only() -> bool:
    return os.environ.get(_SQL_ONLY_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _extract_sql(text: str) -> str:
    m = _SQL_BLOCK_RE.search(text)
    return (m.group(1) if m else text).strip().rstrip(";")


def wrap_sql(sql: str, eval_metric: str) -> str:
    tpl = _SQL_WRAPPER_FILE.read_text(encoding="utf-8")
    return tpl.replace("__SQL__", sql.replace(chr(34) * 3, chr(39) * 3)).replace("__METRIC__", eval_metric)


def sql_of(code: str) -> str:
    """The SQL inside a wrapper (shown in the fix prompt); the code itself otherwise."""
    m = _SQL_IN_WRAPPER_RE.search(code or "")
    return m.group(1).strip() if m else (code or "")

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    return (m.group(1) if m else text).strip()


def generate(question: str, schema: str, plan: str, experiences: str,
             eval_metric: str, client: DeepSeekClient,
             *, index_guide: str | None = None,
             symcol: str | None = None) -> str:
    """Write the script. `index_guide` defaults to the composed guide (F1).

    `symcol` is the name of the symbolic column (E5); it defaults to
    `$SONAR_SYMBOL_COLUMN`, i.e. to `sym`.
    """
    prompt = (_INITIAL_SQL if sql_only() else _INITIAL).read_text(encoding="utf-8").format(
        schema=schema,
        experiences=experiences or "(none)",
        question=question,
        plan=plan,
        eval_metric=eval_metric,
        index_guide=(load_index_guide() if index_guide is None else index_guide),
        symcol=resolve_symbol_column(symcol),
    )
    with llm_tag("generator"):   # usage-log label; the call itself is unchanged
        raw = client.chat([{"role": "user", "content": prompt}])
    if sql_only():
        return wrap_sql(_extract_sql(raw), eval_metric)
    return _extract_code(raw)


def regenerate(question: str, schema: str, plan: str, experiences: str,
               eval_metric: str,
               previous_code: str, error_status: str, error_message: str,
               client: DeepSeekClient, *, index_guide: str | None = None,
               symcol: str | None = None) -> str:
    """Fix the script. `index_guide` defaults to the composed guide (F1)."""
    prompt = (_FIX_SQL if sql_only() else _FIX).read_text(encoding="utf-8").format(
        schema=schema,
        experiences=experiences or "(none)",
        question=question,
        plan=plan,
        eval_metric=eval_metric,
        previous_code=(sql_of(previous_code) if sql_only() else previous_code),
        error_status=error_status,
        error_message=error_message,
        index_guide=(load_index_guide() if index_guide is None else index_guide),
        symcol=resolve_symbol_column(symcol),
    )
    with llm_tag("generator_fix"):   # usage-log label; the call is unchanged
        raw = client.chat([{"role": "user", "content": prompt}])
    if sql_only():
        return wrap_sql(_extract_sql(raw), eval_metric)
    return _extract_code(raw)
