"""LLM agents of the streaming cold start.

Each agent owns exactly one prompt file from `prompts/` and is rendered with
`str.format`; the prompt texts are used verbatim and are never edited here.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from sonar_ts.index_guide import (load_index_guide,  # noqa: E402
                                  resolve_symbol_column)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ADDENDUM F1: the runtime facts carry this marker instead of a copy of the
# index guide. It is replaced with the guide the ONLINE pipeline sees, composed
# from `sonar_ts/prompts/` — one source of truth, so a skill can never be
# distilled against a description of the index that the code generator does not
# get.
INDEX_GUIDE_MARKER = "<<INDEX_GUIDE>>"

# ADDENDUM H1: the runtime facts name the symbolic column twice
# (`<<SYMCOL>>_len, <<SYMCOL>>`). The name is never a constant here either: it
# is whatever `sonar_ts.index_guide.resolve_symbol_column` returns, the same
# resolver `eval.py` and `evidence.py` use ("sax" by default, "sym" under
# SONAR_SYMBOL_COLUMN=sym), so the distiller cannot describe a column that the
# task database does not have.
SYMCOL_MARKER = "<<SYMCOL>>"

# ADDENDUM F2: what `{index_notes}` says when the reflection carries no index
# fields. The shape is the same one `orchestrator.render_index_notes` writes,
# so the drafter always reads the same two lines.
NO_INDEX_NOTES = ("index_use: (none)\nindex_signature: (none)\n"
                  "ranking_key: (none)")

_JSON_REMINDER = ("Your previous reply could not be parsed. "
                  "Reply with the JSON object only, nothing else.")

DOC_REMINDER = ("Your previous reply could not be parsed as a skill document. "
                "Reply with the document only, starting with `---` on the very "
                "first line: no code fence, no preamble, no commentary.")

# ADDENDUM C2: a reply that stopped at the token ceiling is not a document, and
# asking for the same thing again would only be cut off in the same place.
LENGTH_REMINDER = ("Your previous reply was cut off. Write a SHORTER document "
                   "that fits.")


class LabelLeak(AssertionError):
    """A rendered prompt carries something only the benchmark labels know."""


def load_runtime_facts(prompts_dir: Path | str = PROMPTS_DIR,
                       symbolization: str | None = None,
                       column: str | None = None) -> str:
    """The runtime facts with the markers resolved (ADDENDUM F1, H1).

    `<<INDEX_GUIDE>>` becomes the composed guide the ONLINE pipeline sees and
    `<<SYMCOL>>` becomes the resolved symbol column name. A runtime-facts file
    without a marker is used as written — the markers are the only thing this
    function knows about the text.
    """
    text = (Path(prompts_dir) / "runtime_facts_sax.md").read_text(
        encoding="utf-8").strip()
    if SYMCOL_MARKER in text:
        text = text.replace(SYMCOL_MARKER, resolve_symbol_column(column))
    if INDEX_GUIDE_MARKER not in text:
        return text
    return text.replace(INDEX_GUIDE_MARKER,
                        load_index_guide(symbolization).strip())


def extract_json(text: str) -> Dict[str, Any] | None:
    """Return the first balanced `{…}` block of `text` parsed as JSON."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if len(s.split("```")) > 1 else s
        if s.startswith("json"):
            s = s[4:]
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = s.find("{", start + 1)
    return None


@dataclass
class CallStats:
    """Thread-safe counter of LLM calls made by the agents."""

    calls: int = 0
    by_agent: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, agent: str) -> None:
        with self._lock:
            self.calls += 1
            self.by_agent[agent] = self.by_agent.get(agent, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"calls": self.calls, "by_agent": dict(self.by_agent)}


@dataclass
class _PlainReply:
    """What `chat_full` returns for a client that only offers `chat`."""

    content: str
    finish_reason: str | None = None


class CountingClient:
    """Transparent proxy that books every `chat()` of a worker client.

    The online pipeline (planner + code generator) talks to its own client, so
    without this wrapper those calls never reach `CallStats` and the reported
    budget is only the distillation agents — a small fraction of the truth.
    """

    def __init__(self, client: Any, stats: CallStats, label: str):
        self._client = client
        self._stats = stats
        self._label = label

    def chat(self, messages: List[Dict[str, str]],
             **kw: Any) -> str:
        self._stats.bump(self._label)
        return self._client.chat(messages, **kw)

    def chat_full(self, messages: List[Dict[str, str]],
                  **kw: Any) -> Any:
        """Book the call like `chat` does; without this the proxy would hand
        the wrapped client's own `chat_full` out through `__getattr__` and the
        call would never reach `CallStats`."""
        self._stats.bump(self._label)
        full = getattr(self._client, "chat_full", None)
        if full is None:   # a client that only speaks `chat`
            return _PlainReply(str(self._client.chat(messages)), None)
        return full(messages, **kw)

    def __getattr__(self, name: str) -> Any:   # temperature, model, …
        return getattr(self._client, name)


def chat_reply(client: Any, messages: List[Dict[str, str]],
               *, max_tokens: int | None = None) -> tuple[str, str | None]:
    """`(text, finish_reason)` from any client (ADDENDUM C2).

    `DeepSeekClient` offers `chat_full`; anything simpler (the test fakes, the
    counting proxy of the pipeline) only has `chat`, and then the finish reason
    is unknown — which the callers treat as "not truncated".
    """
    full = getattr(client, "chat_full", None)
    if full is not None:
        reply = full(messages, max_tokens=max_tokens)
        return str(reply.content), getattr(reply, "finish_reason", None)
    return str(client.chat(messages)), None


class Agents:
    """The eight prompt-driven agents, sharing one client and one counter.

    `guard` is called with every user message *before* it is sent, so the
    label-blindness check covers all agents in one place. `max_tokens` is the
    distillation ceiling of ADDENDUM C2: these agents write whole documents and
    must not be cut off, while the pipeline keeps the configured value.
    """

    def __init__(self, client: Any, *, prompts_dir: Path | str = PROMPTS_DIR,
                 stats: CallStats | None = None,
                 guard: Callable[[str], None] | None = None,
                 max_tokens: int | None = None):
        self.client = client
        self.prompts_dir = Path(prompts_dir)
        self.stats = stats or CallStats()
        self.guard = guard
        self.max_tokens = max_tokens
        self.runtime_facts = load_runtime_facts(self.prompts_dir)

    # ---------------------------------------------------------------- utils
    def _template(self, name: str) -> str:
        return (self.prompts_dir / f"{name}.txt").read_text(encoding="utf-8")

    def _chat(self, agent: str, messages: List[Dict[str, str]]) -> str:
        return self._chat_full(agent, messages)[0]

    def _chat_full(self, agent: str, messages: List[Dict[str, str]]
                   ) -> tuple[str, str | None]:
        if self.guard is not None:
            for m in messages:
                if m.get("role") == "user":
                    self.guard(str(m.get("content", "")))
        self.stats.bump(agent)
        return chat_reply(self.client, messages, max_tokens=self.max_tokens)

    def _chat_json(self, agent: str, prompt: str
                   ) -> tuple[Dict[str, Any] | None, List[str]]:
        """One call, one retry if the answer does not contain JSON."""
        obj, raws, _reasons = self._chat_json_full(agent, prompt)
        return obj, raws

    def _chat_json_full(self, agent: str, prompt: str
                        ) -> tuple[Dict[str, Any] | None, List[str],
                                   List[str | None]]:
        """`_chat_json` plus the `finish_reason` of each attempt.

        A caller that must tell "the model said this" from "the reply was cut
        off at max_tokens" needs the reason: `extract_json` returns `None` for
        both, and a default verdict built on `None` is then indistinguishable
        from a real one.
        """
        raws: List[str] = []
        reasons: List[str | None] = []
        raw, reason = self._chat_full(agent, [{"role": "user",
                                               "content": prompt}])
        raws.append(raw)
        reasons.append(reason)
        obj = extract_json(raw)
        if obj is not None:
            return obj, raws, reasons
        raw2, reason2 = self._chat_full(agent, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _JSON_REMINDER},
        ])
        raws.append(raw2)
        reasons.append(reason2)
        return extract_json(raw2), raws, reasons

    # --------------------------------------------------------------- agents
    def group(self, *, manifest: str, questions: str
              ) -> tuple[Dict[str, Any] | None, str, List[str]]:
        """Sort the unrouted failures into groups one skill each could serve."""
        prompt = self._template("grouper").format(
            manifest=manifest,
            questions=questions,
        )
        obj, raws = self._chat_json("grouper", prompt)
        return obj, prompt, raws

    def reflect(self, *, skill_in_use: str, manifest: str,
                failed_cases: str, successful_cases: str,
                schema_example: str = "(unavailable)"
                ) -> tuple[Dict[str, Any] | None, str, List[str]]:
        prompt = self._template("reflector").format(
            runtime_facts=self.runtime_facts,
            skill_in_use=skill_in_use,
            manifest=manifest,
            schema_example=schema_example,
            failed_cases=failed_cases,
            successful_cases=successful_cases,
        )
        obj, raws = self._chat_json("reflector", prompt)
        return obj, prompt, raws

    def curate(self, *, manifest: str, skill_in_use_id: str,
               diagnosis_json: str, example_questions: str
               ) -> tuple[Dict[str, Any] | None, str, List[str]]:
        prompt = self._template("curator").format(
            manifest=manifest,
            skill_in_use_id=skill_in_use_id,
            diagnosis_json=diagnosis_json,
            example_questions=example_questions,
        )
        obj, raws = self._chat_json("curator", prompt)
        return obj, prompt, raws

    def draft(self, *, mode: str, skill_id: str, description: str,
              eval_metrics: str, question_intent: str, error_patterns: str,
              success_notes: str, current_skill: str,
              example_questions: str,
              schema_example: str = "(unavailable)",
              reference_solutions: str = "(none)",
              index_notes: str = NO_INDEX_NOTES,
              previous_reply: str | None = None,
              reminder: str = DOC_REMINDER) -> tuple[str, str, str | None]:
        """One drafter call; with `previous_reply` it is the *retry* turn.

        `index_notes` (ADDENDUM F2) is the reflector's verdict on how the
        failed code used the symbolic index. It has a default so that no call
        site of the drafter — CREATE, PATCH, lint repair, regression retry —
        can raise `KeyError` when a diagnosis carries no index fields.
        """
        prompt = self._template("drafter").format(
            runtime_facts=self.runtime_facts,
            mode=mode,
            skill_id=skill_id,
            description=description,
            eval_metrics=eval_metrics,
            question_intent=question_intent,
            error_patterns=error_patterns,
            success_notes=success_notes,
            current_skill=current_skill,
            example_questions=example_questions,
            schema_example=schema_example,
            reference_solutions=reference_solutions,
            index_notes=index_notes,
        )
        messages = [{"role": "user", "content": prompt}]
        if previous_reply is not None:
            messages += [{"role": "assistant", "content": previous_reply},
                         {"role": "user", "content": reminder}]
        raw, finish = self._chat_full("drafter", messages)
        return raw, prompt, finish

    def refine(self, *, mode: str, skill_id: str, description: str,
               eval_metrics: str, primary_skill: str, other_skills: str,
               schema_example: str, exam_results: str,
               reference_solutions: str,
               previous_reply: str | None = None,
               reminder: str = DOC_REMINDER) -> tuple[str, str, str | None]:
        """MERGE / REFINE over everything one skill is responsible for (C6)."""
        prompt = self._template("refiner").format(
            runtime_facts=self.runtime_facts,
            mode=mode,
            skill_id=skill_id,
            description=description,
            eval_metrics=eval_metrics,
            primary_skill=primary_skill,
            other_skills=other_skills,
            schema_example=schema_example,
            exam_results=exam_results,
            reference_solutions=reference_solutions,
        )
        messages = [{"role": "user", "content": prompt}]
        if previous_reply is not None:
            messages += [{"role": "assistant", "content": previous_reply},
                         {"role": "user", "content": reminder}]
        raw, finish = self._chat_full("refiner", messages)
        return raw, prompt, finish

    def cluster(self, *, skills: str
                ) -> tuple[Dict[str, Any] | None, str, List[str]]:
        """Which skills do the same job and could be one document (C6)."""
        obj, prompt, raws, _reasons = self.cluster_full(skills=skills)
        return obj, prompt, raws

    def cluster_full(self, *, skills: str
                     ) -> tuple[Dict[str, Any] | None, str, List[str],
                                List[str | None]]:
        """`cluster` plus each attempt's `finish_reason` (see `_chat_json_full`)."""
        prompt = self._template("cluster").format(skills=skills)
        obj, raws, reasons = self._chat_json_full("cluster", prompt)
        return obj, prompt, raws, reasons

    def critique(self, *, manifest: str, draft_skill: str,
                 example_cases: str) -> str:
        prompt = self._template("critic").format(
            runtime_facts=self.runtime_facts,
            manifest=manifest,
            draft_skill=draft_skill,
            example_cases=example_cases,
        )
        return self._chat("critic", [{"role": "user", "content": prompt}])

    def lint(self, *, skill_text: str) -> tuple[Dict[str, Any] | None, List[str]]:
        prompt = self._template("lint").format(skill_text=skill_text)
        obj, raws = self._chat_json("lint", prompt)
        return obj, raws
