"""Token and time accounting for the pipeline (off unless asked for).

Two independent things live here:

* a per-run tally that `pipeline.run_task` opens with `begin_task` and closes
  with `end_task`; `llm.DeepSeekClient.chat_full` adds every successful call
  to it, so the record of one task can report its own calls/tokens/LLM time.
  The tally is always kept (it is a few integer additions) and also summed
  per task id for the whole process (`task_totals`);
* an optional JSONL log: when the environment variable `SONAR_USAGE_LOG`
  names a file, one short JSON line per LLM call and per code execution is
  appended to it.

Nothing in this module may change what the pipeline does: every function that
is called from the hot path swallows its own errors.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

ENV_VAR = "SONAR_USAGE_LOG"

_COUNTERS = ("llm_calls", "prompt_tokens", "completion_tokens", "llm_time_s")


class TaskUsage:
    """Counters of one `run_task` call."""

    def __init__(self, task_id: Optional[str]):
        self.task_id = task_id
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_time_s = 0.0
        self._lock = threading.Lock()

    def add(self, prompt_tokens: Any, completion_tokens: Any,
            latency_s: float) -> None:
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += _as_int(prompt_tokens)
            self.completion_tokens += _as_int(completion_tokens)
            self.llm_time_s += float(latency_s)


_CURRENT: contextvars.ContextVar[Optional[TaskUsage]] = contextvars.ContextVar(
    "sonar_ts_task_usage", default=None)
_TAG: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "sonar_ts_llm_tag", default=None)

_TOTALS: Dict[str, Dict[str, float]] = {}
_TOTALS_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_warned = False


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


# -- task context ------------------------------------------------------------

def begin_task(task_id: Optional[str]) -> contextvars.Token:
    """Make `task_id` the current task of this thread/context."""
    return _CURRENT.set(TaskUsage(task_id))


def end_task(token: contextvars.Token) -> None:
    try:
        _CURRENT.reset(token)
    except Exception:  # noqa: BLE001  (a token from another context)
        _CURRENT.set(None)


def current_usage() -> Optional[TaskUsage]:
    return _CURRENT.get()


def current_task_id() -> Optional[str]:
    usage = _CURRENT.get()
    return usage.task_id if usage is not None else None


def task_totals(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Process-wide sums per task id (all runs of that id so far)."""
    with _TOTALS_LOCK:
        if task_id is not None:
            return dict(_TOTALS.get(task_id, {}))
        return {k: dict(v) for k, v in _TOTALS.items()}


# -- stage tag ---------------------------------------------------------------

@contextlib.contextmanager
def llm_tag(tag: Optional[str]) -> Iterator[None]:
    """Label the LLM calls made inside the block (e.g. "planner").

    Used instead of a `tag=` keyword so that duck-typed clients whose `chat`
    takes only `messages` (test fakes, wrappers) keep working unchanged.
    """
    token = _TAG.set(tag)
    try:
        yield
    finally:
        _TAG.reset(token)


def current_tag() -> Optional[str]:
    return _TAG.get()


# -- recording ---------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _write_line(record: Dict[str, Any]) -> None:
    global _warned
    path = os.environ.get(ENV_VAR, "").strip()
    if not path:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with _WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:  # noqa: BLE001
        if not _warned:
            _warned = True
            try:
                sys.stderr.write(f"[usage-log] cannot write {path}: "
                                 f"{type(exc).__name__}: {exc}\n")
            except Exception:  # noqa: BLE001
                pass


def record_llm_call(*, body: Any, model: str, tag: Optional[str],
                    latency_s: float, attempts: int,
                    finish_reason: Optional[str]) -> None:
    """Book one successful chat completion. Never raises."""
    try:
        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        cache_hit = usage.get("prompt_cache_hit_tokens")
        if cache_hit is None:   # OpenAI-style servers
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cache_hit = details.get("cached_tokens")
        effective_tag = tag if tag is not None else _TAG.get()
        task = _CURRENT.get()
        task_id = task.task_id if task is not None else None

        if task is not None:
            task.add(prompt_tokens, completion_tokens, latency_s)
        if task_id is not None:
            with _TOTALS_LOCK:
                tot = _TOTALS.setdefault(task_id, {k: 0 for k in _COUNTERS})
                tot["llm_calls"] += 1
                tot["prompt_tokens"] += _as_int(prompt_tokens)
                tot["completion_tokens"] += _as_int(completion_tokens)
                tot["llm_time_s"] += float(latency_s)

        if not os.environ.get(ENV_VAR, "").strip():
            return
        _write_line({
            "kind": "llm",
            "ts": _now_iso(),
            "task_id": task_id,
            "tag": effective_tag,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cache_hit_tokens": cache_hit,
            "latency_s": round(float(latency_s), 4),
            "attempts": attempts,
            "finish_reason": finish_reason,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        })
    except Exception:  # noqa: BLE001
        pass


def record_exec(*, task_id: Optional[str], status: str,
                elapsed_s: float, sql_s: float = 0.0) -> None:
    """Book one run of generated code. Never raises."""
    try:
        if not os.environ.get(ENV_VAR, "").strip():
            return
        _write_line({
            "kind": "exec",
            "ts": _now_iso(),
            "task_id": task_id,
            "status": status,
            "elapsed_s": round(float(elapsed_s), 4),
            "sql_s": round(float(sql_s), 4),
        })
    except Exception:  # noqa: BLE001
        pass


def task_timing(usage: Optional[TaskUsage], *, wall_s: float,
                exec_elapsed: list, exec_sql: Optional[list] = None) -> Dict[str, Any]:
    """The `timing` block of a run record. Never raises."""
    try:
        return {
            "wall_s": round(float(wall_s), 4),
            "llm_calls": usage.llm_calls if usage else 0,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "llm_time_s": round(usage.llm_time_s, 4) if usage else 0.0,
            "exec_time_s": round(float(sum(float(x) for x in exec_elapsed)), 4),
            "exec_runs": len(exec_elapsed),
            "sql_time_s": round(float(sum(float(x) for x in (exec_sql or []))), 4),
        }
    except Exception:  # noqa: BLE001
        return {}
