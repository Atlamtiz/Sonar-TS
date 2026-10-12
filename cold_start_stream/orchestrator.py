"""Streaming, label-free cold start: solve a batch, group, distil, gate, commit.

ADDENDUM C turns the sequential v1 procedure into a batch-synchronous one: all
triggers of a batch are distilled CONCURRENTLY against the library snapshot
taken when the batch's distillation phase starts, and their commits are then
applied in a deterministic order (C5). A patch that regresses the replay set no
longer forks (C1) — it gets one regression-aware retry and is otherwise
rejected — and the library is kept inside a budget by a cluster → merge →
refine consolidation (C6).
"""

from __future__ import annotations

import hashlib
import json
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from sonar_ts import planner as plan_mod
from sonar_ts._paths import db_path_for
from sonar_ts.evaluator import score_one
from sonar_ts.executor import EXEC_CONCURRENCY
from sonar_ts.pipeline import run_task as _default_run_task
from sonar_ts.schema import build_schema_text
from sonar_ts.skills import SkillLibrary, parse_skill_text

from .agents import (Agents, CallStats, CountingClient, DOC_REMINDER,
                     LabelLeak, LENGTH_REMINDER, PROMPTS_DIR)
from .config import StreamConfig
from .evidence import build_evidence
from .lint import mechanical_lint
from .stream import StreamItem, scrub

NONE_KEY = "NONE"
# `exec_status` values that mean the run produced no measurement at all: the
# pipeline itself died (an API outage, a crashed worker), so the stored 0.0 is
# not evidence about any skill. A resume re-solves these items instead of
# reading them as failures (see `_rollback_internal_failures`).
INTERNAL_FAILURES = ("pipeline_error", "pipeline_crash")
# ADDENDUM D5: items whose PATCH was rejected leave the skill's pool for this
# one. It is grouped together with `NONE` — the questions need a method, and
# the only thing known about them is which skill has already failed to absorb
# them — but the origin is kept, and a group that is mostly one origin becomes
# a SPLIT candidate: a narrower skill instead of yet another patch.
SPLIT_KEY = "SPLIT"
GROUP_KEYS = (NONE_KEY, SPLIT_KEY)
# Which PATCH rejections send their items to the SPLIT pool: only the ones
# where a real candidate was measured and lost, never a draft that never
# existed (unparsable, lint).
SPLIT_REASONS = frozenset({"gate", "replay_regression",
                           "regression_retry_failed"})
# The mechanical-lint rules that really are "something out of THIS data set
# leaked into the skill". Everything else (`too_long`, `unclosed_fence`,
# `no_result`, `snippet_syntax`, `other_skill_ref`, the LLM lint) is a
# mechanical defect, and the repair prompt must say so (D3).
# `literal_sax_string` (F4) belongs here too: a copied index string is one
# window of this data set written into a document that must generalise.
DATASET_SPECIFIC_RULES = frozenset({"literal_datetime", "literal_year",
                                    "training_channel", "question_literal",
                                    "literal_sax_string"})
_KEBAB_RE = re.compile(r"[^a-z0-9]+")

# ADDENDUM F2: what the reflector says about the failed code's use of the
# symbolic index. The enum is the reflector prompt's own list; an answer that
# is not in it is still kept (tolerant parsing) — the drafter reads prose.
INDEX_USE_VALUES = ("searched_ok", "searched_too_strict", "searched_wrong_table",
                    "not_searched_but_should", "not_needed")
INDEX_SIGNATURE_MAX_CHARS = 600

# ADDENDUM H2: the reflector's third index field — for ranked-list answers, the
# quantity and the part of the window it must be measured on. Parsed as
# tolerantly as the other two and rendered as a third `{index_notes}` line, so
# every drafter call site (CREATE, PATCH, the C1 regression retry, the G2
# revision rounds) carries it without a change of its own.
RANKING_KEY_MAX_CHARS = 600

CRITIC_HEADING = "CRITIC FEEDBACK — apply every FIX"
REGRESSION_HEADING = (
    "REGRESSIONS — the current skill answered these correctly and your draft "
    "answers them wrongly; keep your fix and restore them:")
RETIRED_DIR = "_retired"

_FENCE_OPEN_RE = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n")
_FRONTMATTER_LINE_RE = re.compile(r"(?m)^---[ \t]*$")
_VERSION_RE = re.compile(r"^v(\d+)$")


# --------------------------------------------------------------------------- helpers
def kebab(text: str) -> str:
    """Lower-case kebab-case id from arbitrary text."""
    s = _KEBAB_RE.sub("-", str(text).strip().lower()).strip("-")
    return s or "skill"


def truncate_code(code: str, max_lines: int) -> str:
    lines = (code or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"... [{len(lines) - max_lines} more lines]"])


def one_line(text: str) -> str:
    return " ".join(str(text).split())


def _section(body: str, heading: str) -> str:
    """The text under a markdown heading (`## When to use`), one line."""
    m = re.search(rf"(?mi)^#+[ \t]*{re.escape(heading)}[ \t]*$", body or "")
    if m is None:
        return ""
    rest = body[m.end():]
    nxt = re.search(r"(?m)^#+[ \t]*\S", rest)
    return one_line(rest[:nxt.start()] if nxt else rest)[:600]


def strip_wrapper(text: str) -> str:
    """Drop a code fence and any preamble in front of the frontmatter.

    The packaged loader stays strict on purpose; only drafts coming from an
    LLM are unwrapped here, because fencing the answer is the single most
    common way a model breaks the "nothing before or after" instruction.
    """
    s = (text or "").lstrip("﻿").strip()
    m = _FENCE_OPEN_RE.match(s)
    if m is not None:
        s = s[m.end():]
        close = s.rfind("```")
        if close != -1:
            s = s[:close]
        s = s.strip()
    if not s.startswith("---"):
        first = _FRONTMATTER_LINE_RE.search(s)
        if first is not None:
            s = s[first.start():]
    return s.strip()


def normalize_doc(text: str, skill_id: str, description: str) -> str | None:
    """Rewrite a draft with a canonical frontmatter; None if it does not parse."""
    parsed = parse_skill_text(strip_wrapper(text))
    if parsed is None:
        return None
    desc = one_line(parsed.description) or one_line(description)
    return f"---\nid: {skill_id}\ndescription: {desc}\n---\n\n{parsed.body.strip()}\n"


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------- F2: the index fields
def index_fields(diag: Dict[str, Any] | None) -> Tuple[str, str, str]:
    """`(index_use, index_signature, ranking_key)`, parsed tolerantly.

    Missing, null, empty or non-string answers become `""`; a list (a model
    that answered with one item) is joined. `index_use` is lower-cased when it
    names one of the enum values, and otherwise kept as written — the value is
    shown to the drafter as prose, so an unexpected wording is information,
    not an error. `ranking_key` (ADDENDUM H2) is treated exactly like
    `index_signature`.
    """
    if not isinstance(diag, dict):
        return "", "", ""

    def _text(value: Any) -> str:
        if value is None or isinstance(value, bool):
            return ""
        # H2: a JSON object is not prose — `str({})` would put a literal `{}`
        # into the drafter's notes, which reads as a lost placeholder.
        if isinstance(value, dict):
            return ""
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value if v is not None)
        return one_line(str(value)).strip()

    use = _text(diag.get("index_use"))
    if use.lower() in INDEX_USE_VALUES:
        use = use.lower()
    signature = _text(diag.get("index_signature"))[:INDEX_SIGNATURE_MAX_CHARS]
    ranking_key = _text(diag.get("ranking_key"))[:RANKING_KEY_MAX_CHARS]
    return use, signature, ranking_key


def render_index_notes(diag: Dict[str, Any] | None) -> str:
    """The `{index_notes}` value of the drafter prompt (ADDENDUM F2, H2)."""
    use, signature, ranking_key = index_fields(diag)
    return (f"index_use: {use or '(none)'}\n"
            f"index_signature: {signature or '(none)'}\n"
            f"ranking_key: {ranking_key or '(none)'}")


def _map_parallel(fn: Callable[[Any], Any], jobs: Sequence[Any],
                  workers: int, *, name: str = "cs-par") -> List[Any]:
    """`[fn(j) for j in jobs]` on up to `workers` threads, order preserved.

    Plain threads, like the rest of the package; the results keep the input
    order so that the caller's deterministic ordering survives (C5).
    """
    jobs = list(jobs)
    results: List[Any] = [None] * len(jobs)
    errors: List[BaseException] = []
    index = iter(range(len(jobs)))
    lock = threading.Lock()

    def _run() -> None:
        while True:
            with lock:
                i = next(index, None)
            if i is None:
                return
            try:
                results[i] = fn(jobs[i])
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                with lock:
                    errors.append(exc)
                # Keep going: one failed job must not silently leave the rest
                # of the batch unprocessed (the first error is still raised).
                continue

    threads = [threading.Thread(target=_run, name=f"{name}{i}", daemon=False)
               for i in range(max(1, min(workers, len(jobs))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


@dataclass(frozen=True)
class Trigger:
    """One distillation attempt: a pool key plus the items it is about.

    For a real skill id the trigger is the whole pool of that skill. For the
    `NONE` pool the grouper cuts the pool into intent groups first (ADDENDUM
    A1) and each group becomes its own trigger, with its own trace directory
    (`NONE__g1`, `NONE__g2`, …) and its own gate.
    """

    key: str
    label: str
    uids: List[str]
    intent: str = ""
    # ADDENDUM D5: set on a SPLIT candidate — the skill these questions were
    # answered with and whose patch was already tried and rejected. The
    # reflector sees that skill, the curator is told the patch failed, and the
    # gate's baseline is the score the items got under it.
    split_from: str = ""


def assign_groups(parsed: Dict[str, Any] | None,
                  uids: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """Turn a grouper reply into `(intent, uids)` groups, largest first.

    Every uid lands in exactly one group: ids the grouper invented are dropped,
    ids it repeated count only the first time, and ids it forgot (including all
    of them, when the reply could not be parsed at all) become singletons.
    """
    allowed = list(dict.fromkeys(uids))
    remaining = set(allowed)
    groups: List[Tuple[str, List[str]]] = []
    raw_groups = (parsed or {}).get("groups")
    if isinstance(raw_groups, list):
        for entry in raw_groups:
            if not isinstance(entry, dict):
                continue
            members = [str(u) for u in (entry.get("case_ids") or [])]
            kept = [u for u in dict.fromkeys(members) if u in remaining]
            if not kept:
                continue
            remaining.difference_update(kept)
            groups.append((one_line(entry.get("intent") or ""), kept))
    groups.extend(("", [u]) for u in allowed if u in remaining)
    # Largest first (the cap on triggers per batch keeps the big ones), with
    # the stream order as a deterministic tie-break.
    order = {u: i for i, u in enumerate(allowed)}
    groups.sort(key=lambda g: (-len(g[1]), order.get(g[1][0], 0)))
    return groups


def assign_clusters(parsed: Dict[str, Any] | None,
                    skill_ids: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """Turn a cluster reply into `(shared_intent, ids)` clusters (ADDENDUM C6).

    Same contract as `assign_groups`: every id lands in exactly one cluster,
    invented ids are dropped, repeated ids count once, and an unparsable reply
    means "all singletons" — consolidation then merges nothing.
    """
    allowed = list(dict.fromkeys(skill_ids))
    remaining = set(allowed)
    clusters: List[Tuple[str, List[str]]] = []
    raw = (parsed or {}).get("clusters")
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            members = [str(s) for s in (entry.get("skill_ids") or [])]
            kept = [s for s in dict.fromkeys(members) if s in remaining]
            if not kept:
                continue
            remaining.difference_update(kept)
            clusters.append((one_line(entry.get("shared_intent") or ""), kept))
    clusters.extend(("", [s]) for s in allowed if s in remaining)
    order = {s: i for i, s in enumerate(allowed)}
    clusters.sort(key=lambda c: (-len(c[1]), order.get(c[1][0], 0)))
    return clusters


# --------------------------------------------------------------------------- state
@dataclass
class Proposal:
    """What one trigger decided, before anything is written (ADDENDUM C5).

    The distillation phase of a batch runs every trigger against the same
    library snapshot and produces one of these; the commit phase then applies
    them in a deterministic order, which is what makes the parallel run equal
    to a sequential one.
    """

    trigger: Trigger
    outcome: str = "skipped"          # accepted | rejected | skipped | rerouted
    act: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    skill_id: str = ""
    mode: str = "CREATE"              # CREATE | PATCH
    doc: str = ""
    description: str = ""
    redescribe: List[Dict[str, Any]] = field(default_factory=list)
    gate: Dict[str, Any] | None = None
    gate_uids: List[str] = field(default_factory=list)
    bump_uids: List[str] = field(default_factory=list)
    lint_residual: List[Dict[str, Any]] = field(default_factory=list)
    diag: Dict[str, Any] = field(default_factory=dict)
    coherent: List[str] = field(default_factory=list)
    trace: Path | None = None
    route_after_commit: bool = False

    @property
    def d_fail(self) -> float:
        return float((self.gate or {}).get("d_fail", 0.0))


def ids_fingerprint(task_ids: Iterable[str]) -> str:
    """A short digest of the exact sequence of task ids this stream runs.

    `--resume` restores a cursor, pools and replay sets that belong to ONE id
    set; `n_sequence` and `seed` do not distinguish two id files of the same
    length (e.g. a regenerated `split.json`).
    """
    h = hashlib.sha1()
    for tid in task_ids:
        h.update(str(tid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


@dataclass
class StreamState:
    """Everything `--resume` needs; written to state.json after every batch."""

    cursor: int = 0
    batch_index: int = 0
    n_sequence: int = 0
    seed: int = 0
    # Fingerprint of the stream's task ids: `n_sequence` and `seed` alone let a
    # resume with a DIFFERENT id file of the same length continue into another
    # stream with this one's pools, replay sets and cursor. "" = written by an
    # older run, which cannot be checked.
    ids_fingerprint: str = ""
    final_done: bool = False
    replay: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    replay_seen: Dict[str, int] = field(default_factory=dict)
    fail_pool: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # D5: uid → the skill whose PATCH was rejected for it (`patch_failed_for`).
    split_origin: Dict[str, str] = field(default_factory=dict)
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skill_versions: Dict[str, int] = field(default_factory=dict)
    retired: Dict[str, str] = field(default_factory=dict)   # C6: id → survivor
    counters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cursor": self.cursor, "batch_index": self.batch_index,
            "n_sequence": self.n_sequence, "seed": self.seed,
            "ids_fingerprint": self.ids_fingerprint,
            "final_done": self.final_done,
            "replay": self.replay, "replay_seen": self.replay_seen,
            "fail_pool": self.fail_pool, "split_origin": self.split_origin,
            "records": self.records,
            "skill_versions": self.skill_versions, "retired": self.retired,
            "counters": self.counters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamState":
        return cls(**{k: v for k, v in d.items()
                      if k in {f for f in cls.__dataclass_fields__}})


# --------------------------------------------------------------------------- orchestrator
class StreamOrchestrator:
    """Runs the streaming cold start over one shuffled, unlabeled query log."""

    def __init__(self, *,
                 config: StreamConfig,
                 items: List[StreamItem],
                 out_dir: Path | str,
                 exec_cfg: Dict[str, Any],
                 client_factory: Callable[[int], Any],
                 run_task_fn: Callable[..., Any] = _default_run_task,
                 route_fn: Callable[..., List[str]] | None = None,
                 prompts_dir: Path | str = PROMPTS_DIR,
                 channel_literals: Iterable[str] = (),
                 log: Callable[[str], None] | None = None):
        self.cfg = config
        self.items = items
        self.sequence: List[StreamItem] = [it for _ in range(max(1, config.epochs))
                                           for it in items]
        self.out = Path(out_dir)
        self.exec_cfg = exec_cfg
        self.client_factory = client_factory
        self.run_task_fn = run_task_fn
        self.route_fn = route_fn or self._default_route
        self.channel_literals = set(channel_literals)
        self.log = log or (lambda msg: print(msg, flush=True))

        self.lib_dir = self.out / "library"
        self.history_dir = self.out / "history"
        self.traces_dir = self.out / "traces"
        self.state_path = self.out / "state.json"
        self.log_path = self.out / "stream_log.jsonl"
        for d in (self.lib_dir, self.history_dir, self.traces_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.library = SkillLibrary(self.lib_dir)
        self.stats = CallStats()
        # Every task id of this stream; the only strings a prompt can carry
        # that could not have come from anywhere but the benchmark labels.
        self._task_id_tokens = {it.task_id.lower() for it in items if it.task_id}
        # One `Agents` per thread, each on its own API key (see `agents`).
        self._prompts_dir = prompts_dir
        self._agents_local = threading.local()
        self._agents_index = 0
        self.ids_fingerprint = ids_fingerprint(
            it.task_id for it in self.sequence)
        self.state = StreamState(n_sequence=len(self.sequence),
                                 seed=config.seed,
                                 ids_fingerprint=self.ids_fingerprint)
        self.state.counters = {"llm_calls": 0, "items_seen": 0, "accepted": 0,
                               "rejected": 0, "skipped": 0,
                               "rerouted": 0, "triggers": 0, "errors": 0,
                               "superseded": 0, "consolidations": 0,
                               "merged": 0, "retired": 0, "refined": 0,
                               "splits": 0, "split_candidates": 0,
                               # D1: every pipeline run of this stream — the
                               # SOLVE phase, every gate and every replay —
                               # counted by the executor status it ended on.
                               # `state.records` only keeps the LAST attempt
                               # per item, so the timeouts D1 asks about are
                               # not recoverable from it.
                               "exec_status_counts": {},
                               "pipeline_runs": 0,
                               "redescribe_rejected": 0,
                               "redescribe_rejected_by_rule": {}}
        self.by_uid = {it.uid: it for it in items}
        self._llm_base = 0  # LLM calls made before a resume
        # The distillation phase of a batch runs in parallel (C5); every write
        # to the state or to the library goes through this lock.
        self._lock = threading.RLock()
        # D1: executor-status tallies are hot and must never wait behind (or
        # be taken inside) the library lock.
        self._counts_lock = threading.Lock()
        self._schema_cache: Dict[str, str] = {}

    # --------------------------------------------------------------- agents
    @property
    def agents(self) -> Agents:
        """The distillation agents of the CURRENT thread.

        One shared `Agents` would put every reflector / curator / drafter /
        critic / lint / grouper / cluster / refiner call of the whole run on
        `client_factory(0)`, i.e. on a single API key — while C5 runs up to
        `trigger_workers` triggers (plus the parallel merge and refine plans of
        C6) at the same time. Each thread therefore builds its own `Agents`
        with its own key, exactly as the pipeline's `_worker_pool` does.
        """
        got = getattr(self._agents_local, "agents", None)
        if got is None:
            with self._lock:
                index = self._agents_index
                self._agents_index += 1
            got = Agents(self.client_factory(index),
                         prompts_dir=self._prompts_dir, stats=self.stats,
                         guard=self._assert_label_blind,
                         max_tokens=self.cfg.distill_max_tokens)
            self._agents_local.agents = got
        return got

    # ------------------------------------------------------------- persistence
    def save_state(self) -> None:
        with self._lock:
            self.state.counters["llm_calls"] = self._llm_base + self.stats.calls
            payload = json.dumps(self.state.to_dict(), indent=2,
                                 ensure_ascii=False, default=str)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.state_path)

    def load_state(self) -> bool:
        if not self.state_path.is_file():
            return False
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = StreamState.from_dict(data)
        if state.n_sequence != len(self.sequence) or state.seed != self.cfg.seed:
            raise SystemExit(
                "state.json does not match this stream "
                f"(state: n={state.n_sequence} seed={state.seed}; "
                f"now: n={len(self.sequence)} seed={self.cfg.seed}). "
                "Use the same --limit/--seed/--epochs or start a fresh --out.")
        if state.ids_fingerprint and state.ids_fingerprint != self.ids_fingerprint:
            raise SystemExit(
                "state.json belongs to a different set of task ids "
                f"(state: {state.ids_fingerprint}; now: {self.ids_fingerprint}) "
                f"— same length, different items. Use the same --ids-file "
                f"(and the same --seed/--limit) or start a fresh --out.")
        state.ids_fingerprint = self.ids_fingerprint
        self.state = state
        for name in ("triggers", "errors", "superseded", "consolidations",
                     "merged", "retired", "refined", "splits",
                     "split_candidates"):
            self.state.counters.setdefault(name, 0)
        self._llm_base = int(self.state.counters.get("llm_calls", 0))
        self._rollback_internal_failures()
        self._quarantine_orphan_skills()
        self.lint_existing_library()
        return True

    def _rollback_internal_failures(self) -> List[str]:
        """Un-commit items the checkpoint stored as an infrastructure failure.

        `process_batch` advances the cursor and saves as soon as the SOLVE
        phase returns, so an outage that makes every call in a batch raise
        (2026-09-21: HTTP 402 on every key) is checkpointed as a batch of
        score-0.0 records with `exec_status == "pipeline_error"`. Those are no
        measurement at all, yet a plain `--resume` would keep them: they would
        sit in the fail pool as genuine failures, drive triggers, and count in
        `items_seen` and in the summary's score statistics.

        So the resume drops them — record, pool entries, replay entries and
        their share of `items_seen`/`exec_status_counts` — and rewinds the
        cursor to the earliest of them, which re-solves exactly those items.
        `llm_calls` is left alone: those calls were really attempted.
        """
        bad = {uid for uid, rec in self.state.records.items()
               if str(rec.get("exec_status")) in INTERNAL_FAILURES}
        if not bad:
            return []
        counts = self.state.counters.setdefault("exec_status_counts", {})
        for uid in bad:
            rec = self.state.records.pop(uid)
            status = str(rec.get("exec_status"))
            if counts.get(status):
                counts[status] -= 1
                if not counts[status]:
                    counts.pop(status, None)
            self.state.counters["items_seen"] = max(
                0, int(self.state.counters.get("items_seen", 0)) - 1)
            for pool in self.state.fail_pool.values():
                pool.pop(uid, None)
            self.state.split_origin.pop(uid, None)
        for key, entries in list(self.state.replay.items()):
            kept = [e for e in entries if str(e.get("uid")) not in bad]
            if len(kept) != len(entries):
                self.state.replay[key] = kept
        # The item's own most recent position in the stream, so that a rewind
        # redoes it and not a whole earlier epoch.
        last_seen: Dict[str, int] = {}
        for i, item in enumerate(self.sequence[:self.state.cursor]):
            if item.uid in bad:
                last_seen[item.uid] = i
        if last_seen:
            self.state.cursor = min(last_seen.values())
        dropped = sorted(bad)
        self.log(f"[resume] {len(dropped)} item(s) were checkpointed as an "
                 f"infrastructure failure ({'/'.join(INTERNAL_FAILURES)}) — "
                 f"dropped and re-solved from item {self.state.cursor}: "
                 + ", ".join(dropped[:5])
                 + (f", … (+{len(dropped) - 5})" if len(dropped) > 5 else ""))
        return dropped

    def lint_library_report(self) -> List[Dict[str, Any]]:
        """Which skills currently in the library fail the mechanical lint."""
        report: List[Dict[str, Any]] = []
        for sid in self.library.ids():
            skill = self.library.get(sid)
            if skill is None:
                continue
            text = (f"---\nid: {skill.id}\ndescription: {skill.description}\n"
                    f"---\n\n{skill.body}\n")
            res = self.mech_lint(text, skill_id=sid)
            if res["verdict"] == "fail":
                report.append({"skill_id": sid,
                               "rules": sorted({v["rule"]
                                                for v in res["violations"]}),
                               "violations": res["violations"]})
        return report

    def lint_existing_library(self) -> List[Dict[str, Any]]:
        """Mechanically lint the skills a resumed run inherits (ADDENDUM C2).

        Reporting only: a skill that is already in the library was gated on
        real items and deleting it would throw that evidence away. The report
        says which of them a v2 run would never have accepted.
        """
        ids = self.library.ids()
        report = self.lint_library_report()
        if report:
            (self.out / "resume_lint.json").write_text(
                json.dumps({"n_skills": len(ids), "failing": report},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            self.log(f"[resume] {len(report)} of {len(ids)} library skills fail "
                     f"the mechanical lint: "
                     + ", ".join(f"{r['skill_id']}({'/'.join(r['rules'])})"
                                 for r in report))
        return report

    def mech_lint(self, text: str, *, skill_id: str,
                  questions: Sequence[str] = (),
                  library: SkillLibrary | None = None) -> Dict[str, Any]:
        """`mechanical_lint` with this run's channels, limits and sibling ids."""
        lib = self.library if library is None else library
        others = [s for s in lib.ids() if s != skill_id]
        return mechanical_lint(text, self.channel_literals, questions, others,
                               doc_max_chars=self.cfg.doc_max_chars,
                               snippet_max_lines=self.cfg.snippet_max_lines)

    def _quarantine_orphan_skills(self) -> None:
        """Move library files the checkpoint does not know about out of the way.

        `_commit` writes the file and then checkpoints; a crash in between
        would otherwise leave a skill on disk that the resumed run re-creates
        under a `-2` id.
        """
        orphans = [md for md in sorted(self.lib_dir.glob("*.md"))
                   if md.stem not in self.state.skill_versions]
        if not orphans:
            return
        dst = self.out / "orphans"
        dst.mkdir(parents=True, exist_ok=True)
        for md in orphans:
            md.replace(dst / md.name)
        self.log(f"[resume] quarantined {len(orphans)} untracked skill file(s) "
                 f"→ {dst}")

    # ------------------------------------------------------------- parallel solve
    def _worker_pool(self, jobs: List[Any],
                     fn: Callable[[Any, Any, int], None],
                     *, label: str = "pipeline",
                     max_workers: int | None = None) -> None:
        """Run `fn(job, client, worker_id)` over `jobs` on one thread per key.

        The worker's client is wrapped so that its LLM calls are booked under
        `label` in `self.stats` (otherwise only the distillation agents count).
        """
        if not jobs:
            return
        q: "queue.Queue" = queue.Queue()
        for j in jobs:
            q.put(j)
        budget = self.cfg.workers if max_workers is None else max_workers
        n_workers = max(1, min(budget, len(jobs)))

        def _run(worker_id: int) -> None:
            client = CountingClient(self.client_factory(worker_id),
                                    self.stats, label)
            while True:
                try:
                    job = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    fn(job, client, worker_id)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(f"[worker {worker_id}] "
                                     f"{type(exc).__name__}: {exc}\n")
                finally:
                    q.task_done()

        threads = [threading.Thread(target=_run, args=(i,), name=f"cs-w{i}",
                                    daemon=False) for i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _solve(self, items: List[StreamItem], *,
               forced_skill_ids: List[str] | None = None,
               library: SkillLibrary | None = None,
               max_workers: int | None = None) -> Dict[str, Dict[str, Any]]:
        """Run the online pipeline on `items`; return uid → outcome dict."""
        lib = self.library if library is None else library
        out: Dict[str, Dict[str, Any]] = {}
        lock = threading.Lock()

        def _one(item: StreamItem, client: Any, _wid: int) -> None:
            try:
                rec = self.run_task_fn(item.task, client,
                                       executor_config=self.exec_cfg,
                                       library=lib,
                                       forced_skill_ids=forced_skill_ids)
                prediction = rec.prediction
                status = rec.final_status
                message = rec.final_message
                chosen = list(rec.chosen_skills or [])
                planner_skills = list(getattr(rec, "planner_skills", []) or [])
                code = rec.code_attempts[-1] if rec.code_attempts else ""
                exec_log = rec.executor_log or []
                last = exec_log[-1] if exec_log else {}
            except Exception as exc:  # noqa: BLE001
                prediction, status = "", "pipeline_crash"
                message = f"{type(exc).__name__}: {exc}"
                chosen, planner_skills, code, last = [], [], "", {}
            try:
                score, _br = score_one(item.eval_metric, prediction,
                                       item.ground_truth)
            except Exception:  # noqa: BLE001
                score = 0.0
            with lock:
                out[item.uid] = {
                    "uid": item.uid,
                    "question": item.question,
                    "eval_metric": item.eval_metric,
                    "ground_truth": str(item.ground_truth),
                    "prediction": str(prediction),
                    "score": float(score),
                    "chosen_skills": chosen,
                    "planner_skills": planner_skills,
                    "code": scrub(truncate_code(code, self.cfg.code_max_lines)),
                    "exec_status": str(last.get("status", status)),
                    "exec_message": scrub(last.get("message", message))[:800],
                }
                self._count_exec_status(str(last.get("status", status)))

        self._worker_pool(list(items), _one, max_workers=max_workers)
        return out

    def _count_exec_status(self, status: str) -> None:
        """One pipeline run ended on `status` (D1).

        Called from `_solve`, which runs concurrently in the distillation
        phase. It uses a lock of its own rather than `self._lock`: the caller
        already holds `_solve`'s result lock, and `self._lock` is the one held
        across library reads and writes elsewhere.
        """
        key = status or "(unknown)"
        with self._counts_lock:
            counts = self.state.counters.setdefault("exec_status_counts", {})
            counts[key] = int(counts.get(key, 0)) + 1
            self.state.counters["pipeline_runs"] = int(
                self.state.counters.get("pipeline_runs", 0)) + 1

    def _default_route(self, item: StreamItem, manifest: str,
                       client: Any) -> List[str]:
        """Planner-only call (no code generation) — used by the routing check."""
        schema = build_schema_text(item.task["id"])
        return plan_mod.plan(item.question, schema, manifest, client).skill_ids

    # ------------------------------------------------------------- bookkeeping
    def primary_key(self, chosen: List[str]) -> str:
        known = set(self.library.ids())
        for sid in chosen:
            if sid in known:
                return sid
        return NONE_KEY

    def _add_replay(self, key: str, uid: str, score: float) -> None:
        """Reservoir-capped replay set per skill (Algorithm R)."""
        if key == NONE_KEY:
            return
        pool = self.state.replay.setdefault(key, [])
        seen = self.state.replay_seen.get(key, 0)
        entry = {"uid": uid, "score": float(score)}
        existing = {e["uid"]: i for i, e in enumerate(pool)}
        if uid in existing:
            pool[existing[uid]] = entry
            return
        self.state.replay_seen[key] = seen + 1
        if len(pool) < self.cfg.replay_cap:
            pool.append(entry)
            return
        # Seeded from the checkpointed counters only, so a resumed run makes
        # exactly the same reservoir decisions.
        j = random.Random(f"{self.cfg.seed}:{key}:{seen}").randint(0, seen)
        if j < self.cfg.replay_cap:
            pool[j] = entry

    def _pool(self, key: str) -> Dict[str, int]:
        return self.state.fail_pool.setdefault(key, {})

    def _pool_keys(self, key: str) -> List[str]:
        """Which pools a trigger with this key draws from (ADDENDUM D5).

        A group trigger is planned over the `NONE` and `SPLIT` pools together
        — one grouper call sees both — so its items may sit in either.
        """
        return list(GROUP_KEYS) if key in GROUP_KEYS else [key]

    def _pool_view(self, key: str) -> Dict[str, int]:
        """The pool(s) of `key` as one read-only mapping uid → attempts."""
        keys = self._pool_keys(key)
        if len(keys) == 1:
            return self._pool(keys[0])
        merged: Dict[str, int] = {}
        for k in keys:
            merged.update(self._pool(k))
        return merged

    def _bump_in_pool(self, pool_key: str, uids: Iterable[str]) -> None:
        """Count one attempt in exactly ONE pool."""
        pool = self._pool(pool_key)
        for uid in list(uids):
            if uid not in pool:
                continue
            pool[uid] += 1
            if pool[uid] >= self.cfg.max_attempts_per_item:
                pool.pop(uid, None)
                self.state.split_origin.pop(uid, None)

    def _bump_attempts(self, key: str, uids: Iterable[str]) -> None:
        """Count one attempt for a TRIGGER key (which may span two pools)."""
        uids = list(uids)
        for pool_key in self._pool_keys(key):
            self._bump_in_pool(pool_key, uids)

    def _drop_from_pools(self, uid: str) -> None:
        for pool in self.state.fail_pool.values():
            pool.pop(uid, None)
        self.state.split_origin.pop(uid, None)

    def _move_to_split_pool(self, target: str, uids: Iterable[str]) -> List[str]:
        """A rejected PATCH hands its items to the SPLIT pool (ADDENDUM D5).

        With FORK gone there is no other way for a narrower skill to come out
        of a pool that one skill keeps failing on: the items leave `target`'s
        pool, keep the attempts they have already burnt, and remember which
        skill they were answered with.
        """
        moved: List[str] = []
        pool = self._pool(target)
        split = self._pool(SPLIT_KEY)
        for uid in list(uids):
            if uid not in pool:
                continue      # already solved, or out of attempts
            split[uid] = pool.pop(uid)
            self.state.split_origin[uid] = target
            moved.append(uid)
        return moved

    # ------------------------------------------------------------- main loop
    def run(self, *, resume: bool = False) -> Dict[str, Any]:
        if resume:
            if self.load_state():
                self.log(f"[resume] continuing at item {self.state.cursor}"
                         f"/{len(self.sequence)} "
                         f"(batch {self.state.batch_index})")
            else:
                self.log("[resume] no state.json found — starting fresh")
        t_start = time.time()
        bs = self.cfg.batch_size
        while self.state.cursor < len(self.sequence):
            batch = self.sequence[self.state.cursor:self.state.cursor + bs]
            self.state.batch_index += 1
            # `cursor_after` is applied inside, right after the SOLVE phase, so
            # a failure during distillation never throws the batch's work away.
            self.process_batch(batch, final=False,
                               cursor_after=self.state.cursor + len(batch))
            self.save_state()
        # End of stream: distil what is left with the lower threshold (once),
        # then consolidate the library once (C6).
        if self.state.final_done:
            self.log("[resume] final pass already done — not repeating it")
        else:
            self.state.batch_index += 1
            self.process_batch([], final=True)
            # The library and the checkpoint are already whole here; a failed
            # consolidation must not cost the run its summary.json and force a
            # --resume that repeats the whole final pass.
            try:
                self.consolidate(refine=True, reason="end_of_stream")
            except Exception as exc:  # noqa: BLE001
                self._note_error("end-of-stream consolidation", exc)
            self.state.final_done = True
            self.save_state()
        summary = self.write_summary(seconds=time.time() - t_start)
        return summary

    def process_batch(self, batch: List[StreamItem], *, final: bool,
                      cursor_after: int | None = None) -> None:
        t0 = time.time()
        bi = self.state.batch_index
        mean_score = 0.0
        per_key: Dict[str, int] = {}
        if batch:
            self.log(f"[batch {bi:02d}] solving {len(batch)} items with "
                     f"{len(self.library.ids())} skills…")
            results = self._solve(batch)
            scores: List[float] = []
            for item in batch:
                res = results.get(item.uid)
                if res is None:
                    continue
                key = self.primary_key(res["chosen_skills"])
                res["primary_key"] = key
                res["batch"] = bi
                self.state.records[item.uid] = res
                self.state.counters["items_seen"] += 1
                scores.append(res["score"])
                per_key[key] = per_key.get(key, 0) + 1
                if res["score"] >= self.cfg.pass_thr:
                    self._add_replay(key, item.uid, res["score"])
                    self._drop_from_pools(item.uid)
                elif res["score"] < self.cfg.fail_thr:
                    self._pool(key).setdefault(item.uid, 0)
            mean_score = mean(scores)
            self.log(f"[batch {bi:02d}] mean score {mean_score:.4f}  "
                     f"groups: {per_key}")

        # Checkpoint the pipeline work before any distillation runs.
        if cursor_after is not None:
            self.state.cursor = cursor_after
        self.save_state()

        triggers, groups_log = self.plan_triggers(batch_index=bi, final=final)
        if groups_log:
            self.log(f"[batch {bi:02d}] intent groups: "
                     + "; ".join(f"{g['label']}({g['size']}) {g['intent']}"
                                 for g in groups_log))
        actions = self.run_triggers(triggers, batch_index=bi)

        line = {
            "batch": bi, "final": final, "n": len(batch),
            "mean_score": round(mean_score, 4),
            "per_key": per_key,
            "pool_sizes": {k: len(v) for k, v in self.state.fail_pool.items() if v},
            "groups": groups_log,
            "actions": actions,
            "n_skills": len(self.library.ids()),
            "llm_calls": self._llm_base + self.stats.calls,
            "seconds": round(time.time() - t0, 1),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

        # C6: the library must stay inside its budget; the end-of-stream
        # consolidation runs from `run()`, after the final distillation pass.
        if not final and len(self.library.ids()) > self.cfg.library_budget:
            try:
                self.consolidate(refine=False, reason="over_budget")
            except Exception as exc:  # noqa: BLE001
                self._note_error("budget consolidation", exc)

    # ------------------------------------------------- C5: one batch of triggers
    def run_triggers(self, triggers: Sequence[Trigger], *,
                     batch_index: int) -> List[Dict[str, Any]]:
        """Distil all triggers against one snapshot, then commit in order.

        The distillation phase writes nothing: every trigger reads the library
        as it was when the phase started and returns a `Proposal`. The commit
        phase then applies the proposals sorted by trigger label, which is what
        makes a parallel batch equal to a sequential one, and only then do the
        routing checks run (they need the final manifest).
        """
        if not triggers:
            return []
        self.state.counters["triggers"] += len(triggers)
        snapshot = self._snapshot_library()
        try:
            lib = SkillLibrary(snapshot.name)
            n = len(triggers)
            share = max(1, self.cfg.workers // max(1, n))

            def _plan(trigger: Trigger) -> Proposal:
                try:
                    return self.plan_trigger(trigger, batch_index=batch_index,
                                             library=lib, solve_workers=share)
                except Exception as exc:  # noqa: BLE001 — one trigger, not the run
                    reason = f"{type(exc).__name__}: {exc}"
                    sys.stderr.write(f"[batch {batch_index:02d}] trigger "
                                     f"{trigger.label} failed: {reason}\n")
                    return Proposal(trigger=trigger, outcome="error",
                                    reason=reason,
                                    act={"key": trigger.key,
                                         "label": trigger.label,
                                         "action": "error", "reason": reason},
                                    bump_uids=list(trigger.uids))

            workers = max(1, min(self.cfg.trigger_workers,
                                 self.cfg.max_triggers_per_batch, n))
            if workers == 1:
                proposals = [_plan(t) for t in triggers]
            else:
                proposals = list(_map_parallel(_plan, triggers, workers,
                                               name="cs-trigger"))
        finally:
            snapshot.cleanup()

        proposals.sort(key=lambda p: p.trigger.label)
        self._resolve_conflicts(proposals)
        for p in proposals:
            self.apply_proposal(p)
        self.save_state()
        self._route_after_commits(proposals)
        return [p.act for p in sorted(proposals,
                                      key=lambda p: (-len(p.trigger.uids),
                                                     p.trigger.label))]

    def _snapshot_library(self) -> tempfile.TemporaryDirectory:
        """A frozen copy of the library for one batch's distillation phase."""
        tmp = tempfile.TemporaryDirectory(prefix="cs-snap-")
        for md in self.lib_dir.glob("*.md"):
            shutil.copy2(md, Path(tmp.name) / md.name)
        return tmp

    def _resolve_conflicts(self, proposals: List[Proposal]) -> None:
        """Two accepted patches of one skill: the larger `d_fail` wins (C5).

        The loser's items go back into the pool with no attempt burnt — the
        trigger did produce a candidate, it simply lost a race it did not know
        it was in.
        """
        by_target: Dict[str, List[Proposal]] = {}
        for p in proposals:
            if p.outcome == "accepted" and p.mode == "PATCH":
                by_target.setdefault(p.skill_id, []).append(p)
        for target, group in by_target.items():
            if len(group) < 2:
                continue
            # `proposals` is sorted by label, so the first proposal with the
            # best d_fail is a deterministic winner.
            best = max(p.d_fail for p in group)
            winner = next(p for p in group if p.d_fail == best)
            for p in group:
                if p is winner:
                    continue
                p.outcome = "superseded"
                p.bump_uids = []
                p.route_after_commit = False   # nothing of its was committed
                p.act.update(action="superseded", reason="conflict",
                             superseded_by=winner.trigger.label)
                self.log(f"    {p.trigger.label}: patch of {target} superseded "
                         f"by {winner.trigger.label} "
                         f"(d_fail {p.d_fail:+.4f} < {winner.d_fail:+.4f})")

    def _route_after_commits(self, proposals: Sequence[Proposal]) -> None:
        """Routing checks for everything that was committed, in parallel (C5).

        A routing check is a *report*: the commits it follows are already on
        disk and in the checkpoint. One failed planner or curator call must
        therefore never take the run down with it — it is recorded on the act
        and counted, like a failed trigger.
        """
        todo = [p for p in proposals if p.route_after_commit]
        if not todo:
            return
        share = max(1, self.cfg.workers // len(todo))

        def _one(p: Proposal) -> None:
            # D5: a skill split off from another one has to take the items it
            # was split for; the check is the same, the threshold is named.
            min_hit = (self.cfg.split_routing_min_hit if p.trigger.split_from
                       else 0.6)
            try:
                hit = self._routing_check(p.coherent, expect_id=p.skill_id,
                                          trace=p.trace or self.traces_dir,
                                          allow_repair=True, diag=p.diag,
                                          coherent=p.coherent,
                                          min_hit=min_hit,
                                          max_workers=share)
            except Exception as exc:  # noqa: BLE001 — one check, not the run
                self._note_error(f"routing check for {p.skill_id}", exc)
                p.act["routing_error"] = f"{type(exc).__name__}: {exc}"
                return
            p.act["routing_hit"] = hit

        if len(todo) == 1:
            _one(todo[0])
        else:
            list(_map_parallel(_one, todo,
                               max(1, min(self.cfg.trigger_workers, len(todo))),
                               name="cs-route"))
        self.save_state()

    def _note_error(self, what: str, exc: BaseException) -> None:
        """Count and log a failure that must not end the run."""
        with self._lock:
            self.state.counters["errors"] = int(
                self.state.counters.get("errors", 0)) + 1
        sys.stderr.write(f"[error] {what}: {type(exc).__name__}: {exc}\n")
        self.log(f"    {what} FAILED ({type(exc).__name__}: {exc}) — continuing")

    # ------------------------------------------------------------- triggers
    def patch_threshold(self, key: str) -> int:
        """How many failures a skill's own pool needs to trigger a PATCH (D4).

        A skill with a real replay set can be patched on fewer failures: the
        replay set is what stops the patch from breaking what works, so the
        risk of acting early is already covered. Without that practice the
        threshold stays at the CREATE level.
        """
        if len(self.state.replay.get(key, [])) >= self.cfg.patch_replay_min:
            return self.cfg.trigger_min_fail_patch
        return self.cfg.trigger_min_fail_create

    def plan_triggers(self, *, batch_index: int, final: bool = False
                      ) -> Tuple[List[Trigger], List[Dict[str, Any]]]:
        """Which distillations this batch runs, and the group log line.

        Real skill ids trigger on their whole pool, as before, at the PATCH
        threshold of ADDENDUM D4. The `NONE` and `SPLIT` pools are cut into
        intent groups by one grouper call; only groups that reach the CREATE
        threshold become triggers, the rest stay pooled and are re-grouped
        next batch. In the end-of-stream pass both thresholds are
        `final_min_fail`.
        """
        create_thr = (self.cfg.final_min_fail if final
                      else self.cfg.trigger_min_fail_create)
        skill_triggers: List[Trigger] = []
        for key in sorted(self.state.fail_pool.keys()):
            if key in GROUP_KEYS:
                continue
            thr = (self.cfg.final_min_fail if final
                   else self.patch_threshold(key))
            if len(self._pool(key)) < thr:
                continue
            skill_triggers.append(Trigger(key=key, label=key,
                                          uids=self._recent_first(self._pool(key))))
        skill_triggers.sort(key=lambda t: (-len(t.uids), t.label))

        cap = max(1, self.cfg.max_triggers_per_batch)
        # Keep part of the cap for the intent groups, so that a few crowded
        # skill pools cannot starve the NONE pool batch after batch. With no
        # room left at all the grouper call would only be paid and discarded.
        none_qualifies = len(self._pool_view(NONE_KEY)) >= create_thr
        reserve = cap // 2 if none_qualifies else 0
        run_grouper = none_qualifies and (len(skill_triggers) < cap or reserve)

        groups_log: List[Dict[str, Any]] = []
        group_triggers: List[Trigger] = []
        if run_grouper:
            for label, intent, uids in self.group_none_pool(batch_index=batch_index):
                split_from = self._split_candidate(uids)
                if split_from:
                    label = label.replace(NONE_KEY, SPLIT_KEY, 1)
                entry = {"label": label, "size": len(uids), "intent": intent,
                         "triggered": False}
                if split_from:
                    entry["split_from"] = split_from
                groups_log.append(entry)
                if len(uids) >= create_thr:
                    group_triggers.append(Trigger(key=NONE_KEY, label=label,
                                                  uids=uids, intent=intent,
                                                  split_from=split_from))
            group_triggers.sort(key=lambda t: (-len(t.uids), t.label))
        elif none_qualifies:
            self.log(f"[batch {batch_index:02d}] trigger cap {cap} already "
                     f"filled by skill pools — no grouper call this batch")

        # Largest work items first, with `reserve` slots held for the groups.
        keep_skills = skill_triggers[:max(0, cap - min(reserve, len(group_triggers)))]
        keep_groups = group_triggers[:cap - len(keep_skills)]
        if len(keep_skills) + len(keep_groups) < cap:  # unused reserve goes back
            keep_skills = skill_triggers[:cap - len(keep_groups)]
        triggers = sorted(keep_skills + keep_groups,
                          key=lambda t: (-len(t.uids), t.label))

        dropped = len(skill_triggers) + len(group_triggers) - len(triggers)
        if dropped:
            self.log(f"[batch {batch_index:02d}] "
                     f"{len(skill_triggers) + len(group_triggers)} triggers "
                     f"available — keeping {len(triggers)}")
        self.state.counters["split_candidates"] = int(
            self.state.counters.get("split_candidates", 0)) + sum(
                1 for t in triggers if t.split_from)
        # `triggered` means "in the final trigger list", not "big enough".
        chosen = {t.label for t in triggers}
        for entry in groups_log:
            entry["triggered"] = entry["label"] in chosen
        return triggers, groups_log

    def _recent_first(self, uids: Iterable[str]) -> List[str]:
        """Pool order for the prompts: the most recently seen failure first."""
        def key(uid: str) -> Tuple[int, str]:
            rec = self.state.records.get(uid, {})
            return (int(rec.get("batch", 0)), uid)
        return sorted(uids, key=key, reverse=True)

    def _split_candidate(self, uids: Sequence[str]) -> str:
        """The skill a group is a SPLIT candidate for, or "" (ADDENDUM D5).

        A group qualifies when at least `split_min_share` of its items are
        items whose PATCH for the SAME skill was rejected: the questions then
        are not "one more thing that skill should also do" but a narrower kind
        of their own.
        """
        if not uids:
            return ""
        # A skill that consolidation retired is no longer something to split
        # off from: its items simply go back to being unrouted failures.
        alive = set(self.library.ids())
        counts: Dict[str, int] = {}
        for uid in uids:
            origin = self.state.split_origin.get(uid)
            if origin and origin in alive:
                counts[origin] = counts.get(origin, 0) + 1
        if not counts:
            return ""
        best = max(sorted(counts), key=lambda s: counts[s])
        if counts[best] / len(uids) >= self.cfg.split_min_share:
            return best
        return ""

    def group_none_pool(self, *, batch_index: int
                        ) -> List[Tuple[str, str, List[str]]]:
        """One grouper call over the unrouted failures → `(label, intent, uids)`.

        ADDENDUM D5: the SPLIT pool is grouped together with the NONE pool, in
        the same call. Both hold questions that need a method and carry no
        label; the only difference is the origin the items remember, and that
        is used afterwards, when the groups are known.
        """
        trace = self.traces_dir / f"batch_{batch_index:02d}" / NONE_KEY
        trace.mkdir(parents=True, exist_ok=True)
        pool = self._pool_view(NONE_KEY)
        uids = self._recent_first(pool)[:self.cfg.group_max_items]
        payload = [{"uid": u,
                    "eval_metric": self.state.records.get(u, {}).get("eval_metric", ""),
                    "question": scrub(self.state.records.get(u, {}).get("question", ""))}
                   for u in uids]
        (trace / "grouper_in.json").write_text(
            json.dumps({"pool_size": len(pool),
                        "n_split_pool": len(self._pool(SPLIT_KEY)),
                        "items": payload},
                       indent=2, ensure_ascii=False), encoding="utf-8")

        questions = "\n\n".join(
            f"### case {p['uid']}\neval_metric: {p['eval_metric']}\n"
            f"question: {p['question']}" for p in payload) or "(none)"
        parsed, _prompt, raws = self.agents.group(
            manifest=self.library.manifest(), questions=questions)
        groups = assign_groups(parsed, uids)
        (trace / "grouper_out.json").write_text(
            json.dumps({"parsed": parsed, "raw": raws,
                        "groups": [{"intent": i, "case_ids": u}
                                   for i, u in groups]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        return [(f"{NONE_KEY}__g{n}", intent, members)
                for n, (intent, members) in enumerate(groups, start=1)]

    # ------------------------------------------------------------- rendering
    def _index_evidence(self, uid: str, record: Dict[str, Any]) -> str:
        """What the index rows look like at both answers of a case (F3).

        Best effort by construction: `build_evidence` swallows every failure
        and returns "", and this method adds the one thing it cannot know —
        where the task database is.
        """
        item = self.by_uid.get(uid)
        if item is None:
            return ""
        task_id = str(item.task.get("id", ""))
        if not task_id:
            return ""
        try:
            return scrub(build_evidence(
                db_path=db_path_for(task_id),
                question=str(record.get("question", "")),
                eval_metric=str(record.get("eval_metric", "")),
                expected=item.ground_truth,
                produced=record.get("prediction", "")))
        except Exception:  # noqa: BLE001 — never break a reflection
            return ""

    def _render_cases(self, uids: List[str], *, with_code: bool,
                      with_evidence: bool = False,
                      records: Dict[str, Dict[str, Any]] | None = None) -> str:
        """The distiller's view of a set of cases.

        `records` overrides what "what happened on this item" means: a
        revision round (ADDENDUM G2) reflects on what the CANDIDATE produced,
        not on what the run recorded before the gate. The outcomes of `_solve`
        and the records of `state` have the same shape, so the same renderer
        serves both.
        """
        if not uids:
            return "(none)"
        src = self.state.records if records is None else records
        blocks: List[str] = []
        for uid in uids:
            r = src.get(uid)
            if r is None:
                continue
            parts = [
                f"### case {uid}",
                f"eval_metric: {r['eval_metric']}",
                f"question: {scrub(r['question'])}",
                # C3: the distiller cannot see what "right" is from 600
                # characters of a long list.
                f"expected: {scrub(r['ground_truth'])[:self.cfg.gt_max_chars]}",
                f"produced: {scrub(r['prediction'])[:600]}",
                f"score: {r['score']:.3f}",
                f"skills loaded: {', '.join(r['chosen_skills']) or '(none)'}",
                f"executor: {r['exec_status']} — {r['exec_message'][:400] or '(no message)'}",
            ]
            if with_code:
                parts.append("final code:\n```python\n"
                             + scrub(r["code"]) + "\n```")
            if with_evidence:
                evidence = self._index_evidence(uid, r)
                if evidence:
                    parts.append("INDEX EVIDENCE:\n" + evidence)
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks) if blocks else "(none)"

    @staticmethod
    def _with_intent_hint(rendered_cases: str, intent: str) -> str:
        """Prefix the grouper's intent to the rendered cases (A1).

        It goes into the *value* the reflector prompt is formatted with, never
        into the prompt file, so the prompt stays as written.
        """
        if not intent:
            return rendered_cases
        return (f"(These cases were grouped together because they ask for the "
                f"same thing: {scrub(one_line(intent))})\n\n{rendered_cases}")

    @staticmethod
    def _with_split_note(rendered_questions: str, split_from: str) -> str:
        """Append ADDENDUM D5's note to the curator's example questions.

        Like the intent hint of A1 it goes into the *value* the prompt is
        formatted with, never into the prompt file. The note names a skill id,
        which is a name the curator itself invented — never a task id, a
        sub-task name or any other benchmark label.
        """
        if not split_from:
            return rendered_questions
        return (f"{rendered_questions}\n\n"
                f"NOTE: these questions were answered with skill "
                f"`{split_from}`; patching `{split_from}` for them was already "
                f"tried and rejected.")

    def _example_questions(self, uids: List[str]) -> str:
        out = []
        for uid in uids[:6]:
            r = self.state.records.get(uid)
            if r:
                out.append(f"- {uid}: {scrub(r['question'])}")
        return "\n".join(out) if out else "(none)"

    def _eval_metrics(self, uids: Sequence[str]) -> str:
        ms = sorted({self.state.records[u]["eval_metric"]
                     for u in uids if u in self.state.records})
        return ", ".join(ms) if ms else "(unknown)"

    # ------------------------------------------------- C3: what the distiller sees
    def _schema_example(self, uids: Sequence[str]) -> str:
        """Tables and columns of one representative task database (C3).

        The reflector used to guess which columns exist; with the real schema
        in front of it, a companion column the question points at is visible.
        The text is neutral by construction (`build_schema_text` never prints
        the task id) and is scrubbed anyway.
        """
        for uid in uids:
            item = self.by_uid.get(uid)
            if item is None:
                continue
            task_id = str(item.task.get("id", ""))
            if not task_id:
                continue
            with self._lock:
                hit = self._schema_cache.get(task_id)
            if hit is not None:
                return hit
            try:
                text = scrub(build_schema_text(task_id))
            except Exception:  # noqa: BLE001 — no database here, no schema
                continue
            with self._lock:
                self._schema_cache[task_id] = text
            return text
        return "(unavailable)"

    def _reference_solutions(self, key: str, uids: Sequence[str],
                             *, fallback_metrics: Sequence[str] = ()) -> str:
        """Up to two short scripts that answered questions of this group (C3).

        Sources are this trigger's own successes and the skill's replay set;
        the shortest first, because a short correct script shows the method
        without drowning it.

        On the CREATE path both sources are empty by construction — the
        trigger's items are the ones still sitting in a fail pool (score below
        `fail_thr`) and `state.replay[NONE]` is never written — so a brand-new
        skill would never see a reference at all. `fallback_metrics` then
        widens the search to every passing record of the same eval_metric,
        which is the only label-blind thing that says "the same kind of
        question".
        """
        def _collect(candidate_uids: Iterable[str]
                     ) -> List[Tuple[int, str, Dict[str, Any]]]:
            found: List[Tuple[int, str, Dict[str, Any]]] = []
            for uid in candidate_uids:
                rec = self.state.records.get(uid)
                if rec is None or float(rec.get("score", 0.0)) < self.cfg.pass_thr:
                    continue
                code = str(rec.get("code") or "").strip()
                if not code:
                    continue
                found.append((len(code.splitlines()), uid, rec))
            return found

        pool = list(dict.fromkeys(list(uids)
                                  + [e["uid"] for e in
                                     self.state.replay.get(key, [])]))
        cands = _collect(pool)
        if not cands and fallback_metrics:
            wanted = {str(m) for m in fallback_metrics}
            seen = set(pool)
            cands = _collect(uid for uid, rec in self.state.records.items()
                             if uid not in seen
                             and str(rec.get("eval_metric", "")) in wanted)
        cands.sort(key=lambda c: (c[0], c[1]))
        blocks = []
        for _n, uid, rec in cands[:self.cfg.max_reference_solutions]:
            blocks.append("\n".join([
                f"### reference {uid} (score {float(rec['score']):.3f})",
                f"question: {scrub(rec.get('question', ''))}",
                "code:\n```python\n"
                + scrub(truncate_code(str(rec.get("code") or ""),
                                      self.cfg.code_max_lines)) + "\n```",
            ]))
        return "\n\n".join(blocks) if blocks else "(none)"

    # ------------------------------------------------------------- one trigger
    # (`handle_trigger` — plan a trigger against the LIVE library and apply it
    #  at once — was removed: it had no caller and it was exactly the pre-C5
    #  algorithm, so the "sequential convenience" path ran something different
    #  from what `run_triggers(…, trigger_workers=1)` certifies.)
    def plan_trigger(self, trigger: Trigger, *, batch_index: int,
                     library: SkillLibrary | None = None,
                     solve_workers: int | None = None) -> Proposal:
        """Everything up to the gate — reads state, writes only traces (C5)."""
        lib = self.library if library is None else library
        key, label = trigger.key, trigger.label
        trace = self.traces_dir / f"batch_{batch_index:02d}" / label
        trace.mkdir(parents=True, exist_ok=True)
        act: Dict[str, Any] = {"key": key, "label": label, "action": "none"}
        if trigger.intent:
            act["intent"] = trigger.intent
        # D5: the skill these questions were answered with. For an ordinary
        # trigger that is the pool key; for a SPLIT candidate it is the skill
        # whose patch was rejected, which is what the reflector must see.
        skill_key = trigger.split_from or key
        if trigger.split_from:
            act["split_from"] = trigger.split_from
        prop = Proposal(trigger=trigger, act=act, trace=trace)

        pool = self._pool_view(key)
        # Only this trigger's items: for a NONE group that is the group, for a
        # real skill it is the whole pool.
        group_uids = [u for u in trigger.uids if u in pool]
        if not group_uids:
            # An earlier trigger of the same batch already solved these.
            return self._skip(prop, "group_empty")
        fail_uids = sorted(group_uids,
                           key=lambda u: (self.state.records.get(u, {}).get("score", 0.0), u))
        fail_uids = fail_uids[:self.cfg.max_cases_reflect]
        succ_uids = [e["uid"] for e in self.state.replay.get(skill_key, [])
                     ][:self.cfg.max_success_cases]

        skill = lib.get(skill_key)
        skill_text = ("---\n"
                      f"id: {skill.id}\ndescription: {skill.description}\n---\n\n"
                      f"{skill.body}") if skill else "(none)"
        manifest = lib.manifest()
        schema_example = self._schema_example(group_uids)
        references = self._reference_solutions(
            key, group_uids + succ_uids,
            fallback_metrics=sorted({self.state.records[u]["eval_metric"]
                                     for u in group_uids
                                     if u in self.state.records}))

        reflector_in = {
            "key": key,
            "label": label,
            "split_from": trigger.split_from,
            "intent": trigger.intent,
            "group_case_ids": group_uids,
            "failed_case_ids": fail_uids,
            "successful_case_ids": succ_uids,
            "manifest": manifest,
        }
        (trace / "reflector_in.json").write_text(
            json.dumps(reflector_in, indent=2, ensure_ascii=False), encoding="utf-8")

        diag, prompt, raws = self.agents.reflect(
            skill_in_use=skill_text,
            manifest=manifest,
            schema_example=schema_example,
            failed_cases=self._with_intent_hint(
                self._render_cases(fail_uids, with_code=True,
                                   with_evidence=True), trigger.intent),
            successful_cases=self._render_cases(succ_uids, with_code=True),
        )
        # F2: the two index fields are normalised into the diagnosis itself, so
        # the record on disk and everything downstream see one spelling.
        index_use, index_signature, ranking_key = index_fields(diag)
        if isinstance(diag, dict):
            diag["index_use"] = index_use
            diag["index_signature"] = index_signature
            diag["ranking_key"] = ranking_key
        (trace / "reflector_out.json").write_text(
            json.dumps({"parsed": diag, "raw": raws,
                        "index_use": index_use,
                        "index_signature": index_signature,
                        "ranking_key": ranking_key}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        del prompt  # the guard already ran inside Agents._chat

        if diag is None:
            return self._skip(prop, "reflector_unparsed", bump=fail_uids)
        if index_use:
            act["index_use"] = index_use

        in_group = set(group_uids)
        coherent = [u for u in diag.get("coherent_case_ids") or []
                    if u in pool and u in in_group]
        cause = str(diag.get("cause", ""))
        if cause == "not_a_method_problem" or not coherent:
            return self._skip(prop, cause or "no_coherent_group",
                              bump=coherent or fail_uids)
        prop.diag, prop.coherent = diag, coherent

        # C4: a gate on two or three items decides nothing, so the trigger is
        # NOT run — decided here, right after the reflector, because the gate
        # set no longer changes and everything below (curator, drafter,
        # critic, revision, lint) would be paid for a candidate that can never
        # be gated. The attempt is burnt on the coherent items so that the
        # identical trigger does not come back batch after batch.
        gate_uids = self._gate_uids(key, coherent,
                                    diag.get("unrelated_case_ids") or [],
                                    allowed=group_uids)
        if len(gate_uids) < self.cfg.final_min_fail:
            prop.act["n_gate_fail"] = len(gate_uids)
            return self._skip(prop, "gate_set_too_small", bump=coherent)
        # Known from here on, whichever way the trigger ends: these are the
        # items a rejected PATCH was measured on, and the ones D5 splits off.
        prop.gate_uids = gate_uids

        # ------------------------------------------------------------ curate
        curation, _cp, craws = self.agents.curate(
            manifest=manifest,
            skill_in_use_id=skill_key,
            diagnosis_json=json.dumps(diag, ensure_ascii=False, indent=2),
            example_questions=self._with_split_note(
                self._example_questions(coherent), trigger.split_from),
        )
        (trace / "curator_out.json").write_text(
            json.dumps({"parsed": curation, "raw": craws}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        if curation is None:
            return self._skip(prop, "curator_unparsed", bump=coherent)

        action = str(curation.get("action", "SKIP")).strip().upper()
        redescribe = curation.get("redescribe") or []
        target = curation.get("target_skill_id")
        new_skill = curation.get("new_skill") or {}
        prop.redescribe = list(redescribe)

        if action == "SKIP":
            return self._skip(prop, "curator_skip", bump=coherent,
                              keep_redescribe=True)

        if action == "REROUTE":
            prop.outcome = "rerouted"
            prop.skill_id = str(target or key)
            prop.act.update(action="rerouted")
            prop.route_after_commit = True
            return prop

        # PATCH with a missing target degrades to CREATE.
        mode = "PATCH" if action == "PATCH" else "CREATE"
        if mode == "PATCH" and (not target or lib.get(target) is None):
            mode, target = "CREATE", None

        if mode == "PATCH":
            skill_id = str(target)
            current = lib.get(skill_id)
            description = one_line(new_skill.get("description")
                                   or current.description)
            current_text = (f"---\nid: {current.id}\ndescription: "
                            f"{current.description}\n---\n\n{current.body}")
        else:
            raw_id = (new_skill.get("id") or diag.get("question_intent")
                      or trigger.intent or key)
            skill_id = self._unique_id(kebab(raw_id), library=lib)
            description = one_line(new_skill.get("description")
                                   or diag.get("question_intent")
                                   or trigger.intent or "")
            current_text = "(none)"
        prop.mode, prop.skill_id, prop.description = mode, skill_id, description

        # ------------------------------------------------- draft → critic → revise
        candidate = self._draft_cycle(
            mode=mode, skill_id=skill_id, description=description,
            diag=diag, coherent=coherent, current_text=current_text,
            manifest=manifest, trace=trace, schema_example=schema_example,
            reference_solutions=references)
        if candidate is None:
            return self._reject(prop, "draft_unparsable", bump=coherent)

        # ------------------------------------------------------------- lint
        questions = self._questions_of(group_uids)
        candidate, lint_ok, lint_report = self._lint_cycle(
            candidate, skill_id=skill_id, description=description,
            diag=diag, coherent=coherent, trace=trace,
            questions=questions, library=lib, schema_example=schema_example,
            reference_solutions=references)
        (trace / "lint.json").write_text(
            json.dumps(lint_report, indent=2, ensure_ascii=False),
            encoding="utf-8")
        prop.lint_residual = list(lint_report.get("llm_residual") or [])
        if prop.lint_residual:
            act["llm_residual"] = list(prop.lint_residual)
        if not lint_ok:
            return self._reject(prop, "lint_failed", bump=coherent)

        # ------------------------------------------------------------- gate
        gate = self._gate(candidate, skill_id=skill_id, mode=mode,
                          patch_target=(skill_id if mode == "PATCH" else None),
                          fail_uids=gate_uids, library=lib,
                          max_workers=solve_workers)
        (trace / "gate.json").write_text(
            json.dumps({k: v for k, v in gate.items() if k != "new_records"},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        act.update(d_fail=gate["d_fail"], d_replay=gate["d_replay"], mode=mode,
                   n_gate_fail=len(gate_uids))
        if trigger.split_from:
            # D5: the baseline d_fail is measured against is the score these
            # items got WITH that skill — which is what `state.records` holds,
            # because a rejected patch settles nothing.
            act["gate_baseline_skill"] = trigger.split_from

        accept = gate["d_fail"] >= self.cfg.accept_delta
        regressed = (mode == "PATCH"
                     and gate["d_replay"] < -self.cfg.replay_tol)
        if accept and regressed:
            # ADDENDUM C1: no fork — one regression-aware retry, then reject.
            candidate, gate, retry = self._regression_retry(
                candidate, skill_id=skill_id, description=description,
                diag=diag, coherent=coherent, gate=gate, gate_uids=gate_uids,
                questions=questions, trace=trace, library=lib,
                schema_example=schema_example, reference_solutions=references,
                max_workers=solve_workers)
            act["regression_retry"] = retry
            if gate is None:
                return self._reject(prop, "regression_retry_failed",
                                    bump=coherent)
            act.update(d_fail=gate["d_fail"], d_replay=gate["d_replay"])
            accept = (gate["d_fail"] >= self.cfg.accept_delta
                      and gate["d_replay"] >= -self.cfg.replay_tol)

        # --------------------------------------------- G2: revision rounds
        # Only a candidate that missed the fail set gets them: a replay
        # regression is C1's business and was already retried above.
        if (not accept and gate["d_fail"] < self.cfg.accept_delta
                and int(self.cfg.gate_revision_rounds) > 0):
            revised, candidate, gate, rounds, residual = self._revision_rounds(
                candidate, gate=gate, skill_id=skill_id,
                description=description, mode=mode, coherent=coherent,
                gate_uids=gate_uids, questions=questions, manifest=manifest,
                schema_example=schema_example,
                reference_solutions=references,
                intent=str(diag.get("question_intent") or trigger.intent or ""),
                trace=trace, library=lib, max_workers=solve_workers)
            if rounds:
                act["revision_rounds"] = rounds
            if revised:
                accept = True
                act.update(d_fail=gate["d_fail"], d_replay=gate["d_replay"])
                prop.lint_residual = list(residual)
                if residual:
                    act["llm_residual"] = list(residual)
                elif "llm_residual" in act:
                    act.pop("llm_residual")

        prop.gate, prop.gate_uids, prop.doc = gate, gate_uids, candidate
        if not accept:
            reason = ("gate" if gate["d_fail"] < self.cfg.accept_delta
                      else "replay_regression")
            prop.act["skill_id"] = skill_id
            return self._reject(prop, reason, bump=coherent)

        prop.outcome = "accepted"
        prop.route_after_commit = True
        act.update(action="accepted", skill_id=skill_id)
        return prop

    # ------------------------------------------------------- proposal outcomes
    def _skip(self, prop: Proposal, reason: str, *,
              bump: Sequence[str] = (), keep_redescribe: bool = False
              ) -> Proposal:
        prop.outcome, prop.reason = "skipped", reason
        prop.bump_uids = list(bump)
        if not keep_redescribe:
            prop.redescribe = []
        prop.act.update(action="skipped", reason=reason)
        return prop

    def _reject(self, prop: Proposal, reason: str, *,
                bump: Sequence[str] = ()) -> Proposal:
        prop.outcome, prop.reason = "rejected", reason
        prop.bump_uids = list(bump)
        prop.redescribe = []
        prop.act.update(action="rejected", reason=reason)
        return prop

    def apply_proposal(self, proposal: Proposal) -> None:
        """Write what one trigger decided; the caller fixes the order (C5).

        One proposal is checkpointed as a whole: the skill file, the replay
        additions and the pool removals of the same trigger reach state.json
        together. A crash in between leaves a library file the checkpoint does
        not know about, which `_quarantine_orphan_skills` moves aside on the
        next `--resume` — whereas a checkpoint taken inside `_commit` would
        persist the skill without its `_settle` and make the resumed run
        distil a second, duplicate skill for items the first one already
        solved.
        """
        p = proposal
        with self._lock:
            if p.lint_residual:
                self._record_lint_residuals(p.skill_id, p.lint_residual)
            if p.outcome == "accepted":
                skill_id = p.skill_id
                if p.mode == "CREATE":
                    # Another trigger of the same batch may have taken the id.
                    skill_id = self._unique_id(skill_id)
                doc = p.doc
                if skill_id != p.skill_id:
                    doc = normalize_doc(doc, skill_id, p.description) or doc
                    p.skill_id = skill_id
                    p.act["skill_id"] = skill_id
                self._commit(skill_id, doc)
                p.act["redescribed"] = self._apply_redescribe(
                    p.redescribe, stage="accept")
                self._settle(skill_id, p.gate or {}, p.gate_uids)
                self.state.counters["accepted"] += 1
                if p.trigger.split_from:
                    self.state.counters["splits"] = int(
                        self.state.counters.get("splits", 0)) + 1
                    self.log(f"    split: {skill_id} split off from "
                             f"{p.trigger.split_from}")
            elif p.outcome == "rerouted":
                # D6: a REROUTE whose every description was rejected by the
                # mechanical lint wrote nothing — it is not a reroute.
                done = self._apply_redescribe(p.redescribe, stage="reroute")
                p.act["redescribed"] = done
                if done:
                    self.state.counters["rerouted"] += 1
            elif p.outcome == "superseded":
                self.state.counters["superseded"] += 1
            elif p.outcome == "error":
                self.state.counters["errors"] += 1
                self._bump_attempts(p.trigger.key, p.bump_uids)
                self.log(f"    trigger {p.trigger.label} FAILED "
                         f"({p.reason}) — continuing")
            else:
                p.act["redescribed"] = self._apply_redescribe(
                    p.redescribe, stage=str(p.outcome))
                self._bump_attempts(p.trigger.key, p.bump_uids)
                self.state.counters[p.outcome] += 1
                # D5: a patch that was measured and lost does not leave its
                # items in the skill's pool to be patched again next batch —
                # they go to the SPLIT pool, where a narrower skill can pick
                # them up. The attempt above is already counted, so an item
                # that fails again in the SPLIT pool leaves as before.
                if (p.outcome == "rejected" and p.mode == "PATCH"
                        and p.reason in SPLIT_REASONS and p.skill_id):
                    # D5: "the items of that trigger move from X's pool to
                    # the SPLIT pool" — the whole trigger, not only the
                    # `gate_fail_cap` subset the gate happened to re-run. What
                    # stays behind is a remainder of the SAME pool that has
                    # just been measured and rejected, and the next batch
                    # re-triggers the same PATCH on it.
                    moved = self._move_to_split_pool(
                        p.skill_id,
                        list(p.trigger.uids) or p.gate_uids or p.bump_uids)
                    if moved:
                        p.act["split_pool"] = len(moved)
                        p.act["patch_failed_for"] = p.skill_id
                        self.log(f"    {len(moved)} item(s) moved to the "
                                 f"SPLIT pool (patch of {p.skill_id} rejected)")
            self.save_state()

    # ------------------------------------------------------------- sub-steps
    def _unique_id(self, base: str, *,
                   library: SkillLibrary | None = None) -> str:
        base = kebab(base)
        # Retired ids are off limits too: reusing one puts a live skill into
        # `state.retired` (and into every consolidate record) as if it had
        # been merged away.
        existing = (set((library or self.library).ids())
                    | set(self.state.retired))
        if base not in existing:
            return base
        n = 2
        while f"{base}-{n}" in existing:
            n += 1
        return f"{base}-{n}"

    def _gate_uids(self, key: str, coherent: List[str],
                   unrelated: Iterable[str],
                   allowed: Iterable[str] | None = None) -> List[str]:
        """The failing items the gate re-runs: this trigger's pool, coherent first.

        The reflector only ever sees `max_cases_reflect` cases; d_fail is the
        accept statistic for the whole procedure, so it is measured on the
        group's failing pool up to `gate_fail_cap`. `allowed` restricts it to
        the trigger's own items — an intent group is never judged on questions
        that belong to a different group (A1).
        """
        skip = set(coherent) | {str(u) for u in unrelated or []}
        pool = self._pool_view(key)
        candidates = pool if allowed is None else [u for u in allowed if u in pool]
        rest = sorted((u for u in candidates if u not in skip),
                      key=lambda u: (self.state.records.get(u, {}).get("score", 0.0),
                                     u))
        return (list(coherent) + rest)[:self.cfg.gate_fail_cap]

    @staticmethod
    def _repair_header(violations: Sequence[Dict[str, Any]]) -> str:
        """The sentence the repair bullets hang under (D3).

        "data-set-specific statements that must go" is the right instruction
        for a literal year / date / channel id / question literal, and the
        wrong one for a document that is merely too long or has an unclosed
        fence: it tells the drafter to delete content that is not there
        instead of to shorten. The wording therefore follows the rules that
        actually fired, and anything mixed takes the neutral sentence the
        consolidation path already uses.
        """
        rules = {str(v.get("rule", "llm_belief")) for v in violations}
        if rules and rules <= DATASET_SPECIFIC_RULES:
            return "The skill contains data-set-specific statements that must go:"
        return "The skill breaks mechanical rules that must be fixed:"

    @staticmethod
    def _repair_bullets(violations: Sequence[Dict[str, Any]]) -> str:
        """One bullet per violation for the drafter's ERROR PATTERNS (D3).

        "remove/replace: <quote>" is the right instruction for a literal, and
        the wrong one for a document that is simply too long — there is
        nothing to replace. A length violation therefore states the two
        numbers and what to cut, which is what the drafter can act on.
        """
        lines: List[str] = []
        for v in violations:
            if str(v.get("rule")) == "too_long" and v.get("limit"):
                size, limit = v.get("size"), v.get("limit")
                if str(v.get("over")) == "snippet":
                    lines.append(
                        f"- the snippet is {size} lines; the limit is {limit} "
                        f"lines; keep one compact runnable example, drop "
                        f"variants and commentary")
                else:
                    lines.append(
                        f"- the document is {size} characters; the limit is "
                        f"{limit}; cut prose, merge overlapping rules, keep "
                        f"the snippet")
                continue
            lines.append(f"- remove/replace: {v.get('quote', '')} "
                         f"({v.get('why', '')})")
        return "\n".join(lines)

    def _questions_of(self, uids: Iterable[str]) -> List[str]:
        """The question texts of these items — input to the lint's A2 rule."""
        return [str(self.state.records[u]["question"]) for u in uids
                if u in self.state.records]

    def _draft_doc(self, **kw: Any) -> str | None:
        """One drafter call, then one *different* retry if it does not parse.

        ADDENDUM C2: a reply that stopped at the token ceiling is unparsable by
        definition, however well-formed its beginning looks — the retry then
        asks for a shorter document instead of the same one again.
        """
        skill_id, description = kw["skill_id"], kw["description"]
        raw, _prompt, finish = self.agents.draft(**kw)
        truncated = finish == "length"
        doc = None if truncated else normalize_doc(raw, skill_id, description)
        if doc is not None:
            return doc
        raw2, _prompt, finish2 = self.agents.draft(
            previous_reply=raw,
            reminder=(LENGTH_REMINDER if truncated else DOC_REMINDER), **kw)
        if finish2 == "length":
            return None
        return normalize_doc(raw2, skill_id, description)

    def _draft_cycle(self, *, mode: str, skill_id: str, description: str,
                     diag: Dict[str, Any], coherent: List[str],
                     current_text: str, manifest: str,
                     trace: Path, schema_example: str = "(unavailable)",
                     reference_solutions: str = "(none)",
                     prefix: str = "") -> str | None:
        patterns = "\n".join(f"- {p}" for p in diag.get("error_patterns") or []) \
            or "(none)"
        successes = "\n".join(f"- {s}" for s in
                              diag.get("what_successes_do_right") or []) or "(none)"
        intent = str(diag.get("question_intent") or "")
        examples = self._example_questions(coherent)
        metrics = self._eval_metrics(coherent)

        index_notes = render_index_notes(diag)

        def _draft(error_patterns: str, current: str,
                   draft_mode: str | None = None) -> str | None:
            return self._draft_doc(
                mode=draft_mode or mode, skill_id=skill_id,
                description=description, eval_metrics=metrics,
                question_intent=intent, error_patterns=error_patterns,
                success_notes=successes, current_skill=current,
                example_questions=examples, schema_example=schema_example,
                reference_solutions=reference_solutions,
                index_notes=index_notes)

        draft = _draft(patterns, current_text)
        if draft is None:
            return None
        # G2: a revision round runs the same cycle again, so its traces carry
        # a prefix instead of overwriting the ones of the first candidate.
        (trace / f"{prefix}draft_v1.md").write_text(draft, encoding="utf-8")

        critique = self.agents.critique(
            manifest=manifest, draft_skill=draft,
            example_cases=self._render_cases(coherent[:3], with_code=False))
        (trace / f"{prefix}critic.md").write_text(critique, encoding="utf-8")

        # The revision always runs in PATCH mode: v1 *is* the current skill at
        # this point, and only the PATCH branch tells the model to keep what
        # works and change only what the error patterns (incl. the critic's
        # FIXes) call for.
        revise_patterns = (f"{patterns}\n\n{CRITIC_HEADING}\n{critique.strip()}")
        revised = _draft(revise_patterns, draft, "PATCH")
        if revised is None:
            revised = draft
        (trace / f"{prefix}draft_v2.md").write_text(revised, encoding="utf-8")
        return revised

    def _lint_cycle(self, candidate: str, *, skill_id: str, description: str,
                    diag: Dict[str, Any], coherent: List[str],
                    trace: Path,
                    questions: Sequence[str] = (),
                    library: SkillLibrary | None = None,
                    schema_example: str = "(unavailable)",
                    reference_solutions: str = "(none)",
                    repair_name: str = "draft_v3_lintfix.md"
                    ) -> tuple[str, bool, Dict[str, Any]]:
        """Mechanical lint is a hard gate; the LLM lint is advisory (B2).

        Both kinds of violation buy the candidate the same single repair pass
        through the drafter. After it, a mechanical violation rejects the
        candidate, while a surviving LLM remark is only recorded
        (`llm_residual`) and the candidate goes on to the gate.
        """
        report: Dict[str, Any] = {"passes": [], "n_questions": len(questions),
                                  "llm_residual": []}
        text = candidate
        for attempt in (1, 2):
            mech = self.mech_lint(text, skill_id=skill_id, questions=questions,
                                  library=library)
            llm: Dict[str, Any] | None = None
            if mech["verdict"] == "pass":
                llm, _raws = self.agents.lint(skill_text=text)
                if llm is None:
                    llm = {"violations": [], "verdict": "pass"}
            mech_v = list(mech["violations"])
            llm_v = list((llm or {}).get("violations") or [])
            violations = mech_v + llm_v
            verdict = "fail" if violations else "pass"
            report["passes"].append({"attempt": attempt, "mechanical": mech,
                                     "llm": llm, "verdict": verdict})
            # Which rule fired — mechanical rules name themselves, the LLM lint
            # is one rule ("llm_belief").
            report["rules_fired"] = sorted(
                {str(v.get("rule", "llm_belief")) for p in report["passes"]
                 for v in (list(p["mechanical"]["violations"])
                           + list((p["llm"] or {}).get("violations") or []))})
            if verdict == "pass":
                report["verdict"] = "pass"
                return text, True, report
            if attempt == 2:
                break
            bullets = self._repair_bullets(violations)
            repaired = self._draft_doc(
                mode="PATCH", skill_id=skill_id, description=description,
                eval_metrics=self._eval_metrics(coherent),
                question_intent=str(diag.get("question_intent") or ""),
                error_patterns=(self._repair_header(violations) + "\n"
                                + bullets),
                success_notes="(none)", current_skill=text,
                example_questions=self._example_questions(coherent),
                schema_example=schema_example,
                reference_solutions=reference_solutions,
                index_notes=render_index_notes(diag))
            if repaired is None:
                break
            text = repaired
            (trace / repair_name).write_text(text, encoding="utf-8")

        last = report["passes"][-1]
        if last["mechanical"]["violations"]:
            report["verdict"] = "fail"
            return text, False, report
        # Mechanically clean, LLM remarks left over: advisory only.
        # Advisory only; the audit line is written when the proposal is
        # applied (C5), so that a parallel distillation phase writes nothing.
        report["llm_residual"] = list((last["llm"] or {}).get("violations") or [])
        report["verdict"] = "pass"
        return text, True, report

    def _record_lint_residuals(self, skill_id: str,
                               residual: Sequence[Dict[str, Any]]) -> None:
        """Append surviving LLM-lint remarks to `lint_residuals.jsonl` (B2).

        The version is the one the candidate would carry if the gate accepts
        it — the audit file is written before the gate runs, so a remark on a
        candidate the gate later rejects stays in it too.
        """
        if not residual:
            return
        version = self._next_version(skill_id)
        with (self.out / "lint_residuals.jsonl").open("a", encoding="utf-8") as fh:
            for v in residual:
                fh.write(json.dumps({"skill_id": skill_id, "version": version,
                                     "quote": str(v.get("quote", "")),
                                     "why": str(v.get("why", ""))},
                                    ensure_ascii=False) + "\n")
        self.log(f"    lint: {len(residual)} advisory LLM remark(s) kept for "
                 f"{skill_id} v{version}")

    def _temp_library(self, skill_id: str, doc: str,
                      source: Path | str | None = None
                      ) -> tempfile.TemporaryDirectory:
        tmp = tempfile.TemporaryDirectory(prefix="cs-lib-")
        dst = Path(tmp.name)
        for md in Path(source or self.lib_dir).glob("*.md"):
            shutil.copy2(md, dst / md.name)
        (dst / f"{skill_id}.md").write_text(doc, encoding="utf-8")
        return tmp

    def _gate(self, candidate: str, *, skill_id: str, mode: str,
              patch_target: str | None,
              fail_uids: List[str],
              library: SkillLibrary | None = None,
              max_workers: int | None = None) -> Dict[str, Any]:
        """Method quality only: re-run with the candidate forced on."""
        fail_uids = list(fail_uids)[:self.cfg.gate_fail_cap]
        replay = (self.state.replay.get(patch_target, [])
                  if patch_target else [])
        replay_uids = [e["uid"] for e in replay]

        source = (library or self.library).dir
        tmp = self._temp_library(skill_id, candidate, source)
        try:
            lib = SkillLibrary(tmp.name)
            items = [self.by_uid[u] for u in fail_uids + replay_uids
                     if u in self.by_uid]
            new = self._solve(items, forced_skill_ids=[skill_id], library=lib,
                              max_workers=max_workers)
        finally:
            tmp.cleanup()

        old_fail = [self.state.records[u]["score"] for u in fail_uids
                    if u in self.state.records]
        new_fail = [new[u]["score"] for u in fail_uids if u in new]
        old_replay = [e["score"] for e in replay if e["uid"] in new]
        new_replay = [new[e["uid"]]["score"] for e in replay if e["uid"] in new]

        return {
            "skill_id": skill_id, "mode": mode,
            "fail_uids": fail_uids, "replay_uids": replay_uids,
            "old_fail_mean": round(mean(old_fail), 4),
            "new_fail_mean": round(mean(new_fail), 4),
            "old_replay_mean": round(mean(old_replay), 4),
            "new_replay_mean": round(mean(new_replay), 4),
            "d_fail": round(mean(new_fail) - mean(old_fail), 4),
            "d_replay": (round(mean(new_replay) - mean(old_replay), 4)
                         if replay_uids else 0.0),
            "new_scores": {u: round(r["score"], 4) for u, r in new.items()},
            # ADDENDUM G2: what the candidate actually did on each fail item.
            # A revision round reflects on exactly this, and a reader of
            # `gate.json` can see why a rejected candidate was rejected
            # without re-running anything — `new_records` is dropped before
            # the file is written, so it carries the same facts in short form.
            "fail_outcomes": {
                u: {"score": round(float(new[u]["score"]), 4),
                    "prediction": str(new[u]["prediction"])[:600],
                    "code": new[u]["code"],
                    "exec_status": new[u]["exec_status"],
                    "exec_message": new[u]["exec_message"][:400]}
                for u in fail_uids if u in new},
            # Full outcomes so `_settle` can replace a record as a whole
            # instead of leaving a new score next to the old prediction.
            "new_records": new,
        }

    def _next_version(self, skill_id: str) -> int:
        """One past the highest version on disk *and* in the checkpoint."""
        seen = [int(self.state.skill_versions.get(skill_id, 0))]
        hist = self.history_dir / skill_id
        if hist.is_dir():
            for p in hist.glob("v*.md"):
                m = _VERSION_RE.fullmatch(p.stem)
                if m:
                    seen.append(int(m.group(1)))
        return max(seen) + 1

    def _commit(self, skill_id: str, doc: str) -> bool:
        """Write a new version of a skill; False when nothing changed.

        A commit is deliberately NOT a checkpoint: `_commit` and the `_settle`
        of the same proposal must reach state.json together, or a resume after
        a crash between them re-distils items an already-committed skill
        solved and ships the duplicate under a `-2` id. The caller
        (`apply_proposal`, `_apply_merge`, `_refine_survivors`) checkpoints
        once its whole unit of work is in the state; a library file written
        without a checkpoint is quarantined on resume.
        """
        with self._lock:
            path = self.lib_dir / f"{skill_id}.md"
            # A curator that hands back the description a skill already has
            # must not bump its version and write a duplicate history
            # snapshot — `skill_versions` is reported as "how often this skill
            # really changed" and is what `_age_index` orders by.
            if (skill_id in self.state.skill_versions and path.is_file()
                    and path.read_text(encoding="utf-8") == doc):
                return False
            self._write_atomic(path, doc)
            version = self._next_version(skill_id)
            self.state.skill_versions[skill_id] = version
            hist = self.history_dir / skill_id
            hist.mkdir(parents=True, exist_ok=True)
            self._write_atomic(hist / f"v{version}.md", doc)
        self.log(f"    committed {skill_id} v{version}")
        return True

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        """Write via tmp+replace, as `save_state` does.

        Readers of the library (`_apply_redescribe`, the manifest reads of the
        parallel routing phase) take no lock and `SkillLibrary` re-reads the
        directory on every call, so a plain truncate-then-write lets a reader
        see an empty file and the skill silently vanish from that manifest.
        """
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _settle(self, skill_id: str, gate: Dict[str, Any],
                gated_uids: List[str]) -> None:
        """Move items the candidate now solves into the replay set."""
        solved: List[str] = []
        fail_set = set(gate.get("fail_uids") or [])
        new_records = gate.get("new_records") or {}
        for uid, score in (gate.get("new_scores") or {}).items():
            if uid not in fail_set:
                continue  # replay items keep the score they were recorded with
            old = self.state.records.get(uid)
            if old is not None:
                # Replace the whole record, not only its score: prediction,
                # code and executor status must describe the same run.
                rec = dict(new_records.get(uid) or {})
                merged = dict(old)
                merged.update(rec)
                merged["score"] = float(score)
                merged["primary_key"] = skill_id
                merged["gated_by"] = skill_id
                self.state.records[uid] = merged
            if score >= self.cfg.pass_thr:
                self._add_replay(skill_id, uid, float(score))
                self._drop_from_pools(uid)
                solved.append(uid)
        still = [u for u in gated_uids if u not in solved]
        # One attempt per pool the item really sits in: `_bump_attempts` would
        # expand NONE to NONE+SPLIT and charge a SPLIT item twice (D5).
        for key in list(self.state.fail_pool.keys()):
            self._bump_in_pool(key, [u for u in still
                                     if u in self.state.fail_pool[key]])

    def _apply_redescribe(self, entries: List[Dict[str, Any]],
                          *, stage: str = "") -> List[str]:
        """Write the curator's new descriptions — linted first (ADDENDUM D6).

        Every REROUTE path (trigger-level, curator `redescribe`, the routing
        check's repair, consolidation) lands here, and until v3 none of them
        ran the mechanical lint: a description naming another skill
        (`other_skill_ref`) went straight into the library, which is exactly
        the text the planner reads. One violation buys one more curator call
        with the violation quoted; a second one keeps the old description.

        Returns the ids whose description really changed. A caller counts a
        REROUTE only when that list is non-empty: a reroute whose every
        description was rejected changed nothing, and `rerouted` used to count
        it anyway. Every rejection is written to `redescribe_rejected.jsonl`
        so that D7's "lint rejections by rule" can be recovered from the run.
        """
        written: List[str] = []
        for e in entries or []:
            sid = str(e.get("skill_id", ""))
            desc = one_line(e.get("description", ""))
            if not sid or not desc:
                continue
            ok, violations = self._write_description(sid, desc)
            if ok:
                written.append(sid)
                continue
            if not violations:
                continue
            fixed = self._redescribe_repair(sid, desc, violations)
            if fixed is None:
                # The curator echoed the rejected text, returned no entry for
                # this id, or the call failed. Nothing is written; without
                # this record the run keeps no trace of it at all.
                self._record_redescribe_rejected(
                    sid, desc, violations, stage=stage,
                    outcome="no_repair_offered")
                continue
            ok, violations = self._write_description(sid, fixed)
            if ok:
                written.append(sid)
                continue
            # `_record_redescribe_rejected` logs the line this branch used
            # to log itself.
            self._record_redescribe_rejected(
                sid, fixed, violations, stage=stage,
                outcome="repair_rejected", first_rejected=desc)
        return written

    def _record_redescribe_rejected(self, sid: str, desc: str,
                                    violations: Sequence[Dict[str, Any]], *,
                                    stage: str, outcome: str,
                                    first_rejected: str = "") -> None:
        """Append one dropped redescribe to `redescribe_rejected.jsonl` (D6)."""
        rules = sorted({str(v.get("rule", "")) for v in violations})
        row = {"skill_id": sid, "stage": stage, "outcome": outcome,
               "rules": rules, "rejected_description": desc,
               "first_rejected_description": first_rejected or None,
               "violations": [{"rule": str(v.get("rule", "")),
                               "quote": str(v.get("quote", ""))}
                              for v in violations]}
        with self._lock:
            counters = self.state.counters
            counters["redescribe_rejected"] = int(
                counters.get("redescribe_rejected", 0)) + 1
            by_rule = counters.setdefault("redescribe_rejected_by_rule", {})
            for rule in (rules or ["(none)"]):
                by_rule[rule] = int(by_rule.get(rule, 0)) + 1
            with (self.out / "redescribe_rejected.jsonl").open(
                    "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log(f"    redescribe of {sid} NOT written ({outcome}: "
                 f"{', '.join(rules) or 'no violation'}) — "
                 f"the old description stays")

    def _write_description(self, sid: str, desc: str
                           ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Lint and (if clean) commit one new description.

        Returns `(written, violations)`; `(False, [])` means there was nothing
        to do — no such skill, or the curator echoed the description it was
        given.

        Only the DESCRIPTION is linted, not the rebuilt document. The gate
        exists to stop a bad description reaching the planner, and the repair
        the curator can offer is a new description — so a violation that lives
        in the body (`no_result`, `unclosed_fence`, an `other_skill_ref` in
        the prose, a body that is already over the length limit) can never be
        cleared here and would block every future reroute of that skill
        forever. `other_skill_ref` in particular starts firing on an old body
        the moment a later skill is minted. Body defects are the drafting
        path's business; `lint_library_report` still reports them.

        A description short enough not to matter is the normal case, but the
        one document-level rule the description really can break — pushing the
        whole file over `doc_max_chars` — is checked against the file as it
        stands, so it only fires when this change is what puts it over.
        """
        # Read-modify-write of a library file: another thread's `_commit`
        # must not land between the read and the write.
        with self._lock:
            skill = self.library.get(sid)
            if skill is None:
                return False, []
            if one_line(skill.description) == desc:
                return False, []
            body = skill.body.strip()
            doc = f"---\nid: {skill.id}\ndescription: {desc}\n---\n\n{body}\n"
            old_doc = (f"---\nid: {skill.id}\ndescription: "
                       f"{one_line(skill.description)}\n---\n\n{body}\n")
            violations = list(self.mech_lint(desc, skill_id=sid)["violations"])
            limit = self.cfg.doc_max_chars
            if len(doc) > limit >= len(old_doc):
                violations.append({
                    "quote": f"document length {len(doc)} characters",
                    "why": (f"this description takes the document from "
                            f"{len(old_doc)} to {len(doc)} characters, past "
                            f"the {limit}-character limit"),
                    "rule": "too_long", "over": "document",
                    "size": len(doc), "limit": limit,
                })
            if violations:
                return False, violations
            self._commit(sid, doc)
            return True, []

    def _redescribe_repair(self, sid: str, desc: str,
                           violations: Sequence[Dict[str, Any]]) -> str | None:
        """Ask the curator once more, with the violation quoted (D6).

        Deliberately outside the library lock: this is a network call, and the
        commit phase of the batch must not wait for it.
        """
        payload = {
            "redescribe_violation": [
                {"rule": str(v.get("rule", "")), "quote": str(v.get("quote", "")),
                 "why": str(v.get("why", ""))} for v in violations],
            "rejected_description": desc,
            "reroute_note": (
                f"The description you proposed for '{sid}' breaks a mechanical "
                f"rule and was not written. Return REROUTE with a redescribe "
                f"entry for '{sid}' that says in plain words which questions "
                f"the skill is for and which it is not for, and that avoids "
                f"the quoted text."),
        }
        try:
            curation, _p, _raws = self.agents.curate(
                manifest=self.library.manifest(), skill_in_use_id=sid,
                diagnosis_json=json.dumps(payload, ensure_ascii=False, indent=2),
                example_questions=self._example_questions(
                    self._recent_first(self._pool_view(sid))))
        except Exception as exc:  # noqa: BLE001 — keep the old description
            self._note_error(f"redescribe repair for {sid}", exc)
            return None
        for e in (curation or {}).get("redescribe") or []:
            if str(e.get("skill_id", "")) == sid:
                new = one_line(e.get("description", ""))
                if new and new != desc:
                    return new
        return None

    # ------------------------------------------------- C1: regression-aware retry
    def _regressed_cases(self, gate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Replay items the candidate answers worse than the current skill."""
        new_records = gate.get("new_records") or {}
        out: List[Dict[str, Any]] = []
        for uid in gate.get("replay_uids") or []:
            old = next((e["score"] for e in
                        self.state.replay.get(gate.get("skill_id", ""), [])
                        if e["uid"] == uid), None)
            if old is None:
                old = self.state.records.get(uid, {}).get("score", 0.0)
            rec = new_records.get(uid)
            if rec is None:
                continue
            if float(rec["score"]) < float(old) - 1e-9:
                out.append({"uid": uid, "old": float(old),
                            "new": float(rec["score"]), "record": rec})
        out.sort(key=lambda r: (r["new"] - r["old"], r["uid"]))
        return out

    def _render_regressions(self, regressed: Sequence[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for r in regressed[:self.cfg.max_cases_reflect]:
            base = self.state.records.get(r["uid"], {})
            rec = r["record"]
            blocks.append("\n".join([
                f"### case {r['uid']}",
                f"question: {scrub(base.get('question', ''))}",
                f"expected: {scrub(base.get('ground_truth', ''))[:self.cfg.gt_max_chars]}",
                f"produced by your draft: {scrub(rec.get('prediction', ''))[:600]}",
                f"score: {r['old']:.3f} → {r['new']:.3f}",
                "the draft's code:\n```python\n"
                + scrub(truncate_code(rec.get("code", ""),
                                      self.cfg.code_max_lines)) + "\n```",
            ]))
        return "\n\n".join(blocks) if blocks else "(none)"

    def _regression_retry(self, candidate: str, *, skill_id: str,
                          description: str, diag: Dict[str, Any],
                          coherent: List[str], gate: Dict[str, Any],
                          gate_uids: List[str], questions: Sequence[str],
                          trace: Path, library: SkillLibrary | None,
                          schema_example: str, reference_solutions: str,
                          max_workers: int | None
                          ) -> tuple[str, Dict[str, Any] | None, Dict[str, Any]]:
        """One retry that must keep the fix AND restore the replay set (C1).

        v1 forked here, and every one of those 15 forks became a narrow
        near-duplicate that attracted the planner and had no practice. There is
        exactly one retry: the drafter sees the regressed replay cases, the
        candidate is linted again and re-gated once, and if the replay is still
        broken the patch is rejected.
        """
        regressed = self._regressed_cases(gate)
        info: Dict[str, Any] = {"n_regressed": len(regressed),
                                "d_replay_before": gate["d_replay"]}
        patterns = REGRESSION_HEADING + "\n" + self._render_regressions(regressed)
        retry_doc = self._draft_doc(
            mode="PATCH", skill_id=skill_id, description=description,
            eval_metrics=self._eval_metrics(coherent),
            question_intent=str(diag.get("question_intent") or ""),
            error_patterns=patterns, success_notes="(none)",
            current_skill=candidate,
            example_questions=self._example_questions(coherent),
            schema_example=schema_example,
            reference_solutions=reference_solutions,
            index_notes=render_index_notes(diag))
        if retry_doc is None:
            info["outcome"] = "draft_unparsable"
            return candidate, None, info
        (trace / "draft_v4_regression.md").write_text(retry_doc,
                                                      encoding="utf-8")

        retry_doc, lint_ok, lint_report = self._lint_cycle(
            retry_doc, skill_id=skill_id, description=description, diag=diag,
            coherent=coherent, trace=trace, questions=questions,
            library=library, schema_example=schema_example,
            reference_solutions=reference_solutions,
            repair_name="draft_v4_regression_lintfix.md")
        (trace / "lint_regression.json").write_text(
            json.dumps(lint_report, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if not lint_ok:
            info["outcome"] = "lint_failed"
            return candidate, None, info

        new_gate = self._gate(retry_doc, skill_id=skill_id, mode="PATCH",
                              patch_target=skill_id, fail_uids=gate_uids,
                              library=library, max_workers=max_workers)
        (trace / "gate_regression.json").write_text(
            json.dumps({k: v for k, v in new_gate.items()
                        if k != "new_records"}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        info.update(outcome="regated", d_fail=new_gate["d_fail"],
                    d_replay=new_gate["d_replay"])
        return retry_doc, new_gate, info

    # ------------------------------------------- G2: gate-feedback revisions
    def _revision_reflection(self, candidate: str, gate: Dict[str, Any], *,
                             gate_uids: Sequence[str], manifest: str,
                             schema_example: str, intent: str, trace: Path,
                             round_no: int) -> Dict[str, Any] | None:
        """Reflect on what the CANDIDATE did to the gate's items (ADDENDUM G2).

        The ordinary reflection reads the skill that was in use and the code
        the run produced with it; here the skill in use IS the candidate and
        the code is the code the gate just produced with it — which is the
        only view in which "the pattern I wrote loses the true days" is
        visible at all. The reflector prompt is unchanged.
        """
        outcomes = gate.get("new_records") or {}
        scored = sorted(((float(outcomes[u]["score"]), u) for u in gate_uids
                         if u in outcomes), key=lambda t: (t[0], t[1]))
        still_failing = [u for s, u in scored if s < self.cfg.pass_thr]
        now_right = [u for s, u in sorted(scored, key=lambda t: (-t[0], t[1]))
                     if s >= self.cfg.pass_thr]
        # If the candidate is above `pass_thr` everywhere and still missed the
        # gate (a baseline that was already high), the worst items are still
        # the ones worth looking at.
        fail_cases = (still_failing or [u for _s, u in scored]
                      )[:self.cfg.max_cases_reflect]
        # In that fallback every gate item is above `pass_thr`, so the items
        # shown as FAILED would be shown as SUCCESSFUL too — the reflector
        # would read one case under both headings and be told, of the same
        # code, both that it works and that it does not.
        shown_as_failed = set(fail_cases)
        succ_cases = [u for u in now_right
                      if u not in shown_as_failed][:self.cfg.max_success_cases]

        diag, _prompt, raws = self.agents.reflect(
            skill_in_use=candidate,
            manifest=manifest,
            schema_example=schema_example,
            failed_cases=self._with_intent_hint(
                self._render_cases(fail_cases, with_code=True,
                                   with_evidence=True, records=outcomes),
                intent),
            successful_cases=self._render_cases(succ_cases, with_code=True,
                                                records=outcomes),
        )
        index_use, index_signature, ranking_key = index_fields(diag)
        if isinstance(diag, dict):
            diag["index_use"] = index_use
            diag["index_signature"] = index_signature
            diag["ranking_key"] = ranking_key
        (trace / f"revision_r{round_no}_reflector_out.json").write_text(
            json.dumps({"parsed": diag, "raw": raws,
                        "index_use": index_use,
                        "index_signature": index_signature,
                        "ranking_key": ranking_key,
                        "failed_case_ids": fail_cases,
                        "successful_case_ids": succ_cases}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        return diag if isinstance(diag, dict) else None

    def _revision_rounds(self, candidate: str, *, gate: Dict[str, Any],
                         skill_id: str, description: str, mode: str,
                         coherent: List[str], gate_uids: List[str],
                         questions: Sequence[str], manifest: str,
                         schema_example: str, reference_solutions: str,
                         intent: str, trace: Path,
                         library: SkillLibrary | None,
                         max_workers: int | None
                         ) -> Tuple[bool, str, Dict[str, Any],
                                    List[Dict[str, Any]],
                                    List[Dict[str, Any]]]:
        """Up to `cfg.gate_revision_rounds` rounds of gate feedback (G2).

        A candidate that passed the lint and missed the gate used to be thrown
        away whole: the next trigger for the same intent started from nothing
        and made the same mistake (9 times in the three streams of shape_v5).
        Each round reflects on the candidate's own gate failures, drafts a
        PATCH of it through the usual draft → critic → revise → lint cycle and
        re-gates it on the SAME items. The better of (candidate, revision) by
        `d_fail` is the base of the next round.

        Returns `(accepted, doc, gate, rounds, lint_residual)`. When no round
        is accepted the ORIGINAL candidate and its gate come back unchanged,
        so the rejection is the one the run would have written without this
        feature at all — only `rounds` is added to the action record.
        """
        rounds: List[Dict[str, Any]] = []
        base_doc, base_gate = candidate, gate
        for n in range(1, int(self.cfg.gate_revision_rounds) + 1):
            info: Dict[str, Any] = {"round": n, "d_fail": None,
                                    "d_replay": None, "accepted": False}
            rounds.append(info)
            # A round is an EXTRA attempt at a candidate the run has already
            # decided to reject. A crash inside it must therefore cost the
            # round and nothing else: without this guard the exception leaves
            # `plan_trigger`, `run_triggers` turns the whole trigger into an
            # `error` proposal, and the rejection that was already earned — its
            # reason, its bumped attempts, its SPLIT-pool move (D5) and the
            # `rejected` counter — is replaced by an `errors` count.
            try:
                diag = self._revision_reflection(
                    base_doc, base_gate, gate_uids=gate_uids,
                    manifest=manifest, schema_example=schema_example,
                    intent=intent, trace=trace, round_no=n)
                if diag is None:
                    # Nothing to draft from, and the next round would ask the
                    # same question of the same text.
                    info["outcome"] = "reflector_unparsed"
                    break
                revision = self._draft_cycle(
                    mode="PATCH", skill_id=skill_id, description=description,
                    diag=diag, coherent=coherent, current_text=base_doc,
                    manifest=manifest, trace=trace,
                    schema_example=schema_example,
                    reference_solutions=reference_solutions,
                    prefix=f"revision_r{n}_")
                if revision is None:
                    info["outcome"] = "draft_unparsable"
                    continue
                (trace / f"revision_r{n}_draft.md").write_text(
                    revision, encoding="utf-8")
                revision, lint_ok, lint_report = self._lint_cycle(
                    revision, skill_id=skill_id, description=description,
                    diag=diag, coherent=coherent, trace=trace,
                    questions=questions, library=library,
                    schema_example=schema_example,
                    reference_solutions=reference_solutions,
                    repair_name=f"revision_r{n}_lintfix.md")
                (trace / f"revision_r{n}_lint.json").write_text(
                    json.dumps(lint_report, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                if not lint_ok:
                    info["outcome"] = "lint_failed"
                    continue
                new_gate = self._gate(
                    revision, skill_id=skill_id, mode=mode,
                    patch_target=(skill_id if mode == "PATCH" else None),
                    fail_uids=gate_uids, library=library,
                    max_workers=max_workers)
                (trace / f"revision_r{n}_gate.json").write_text(
                    json.dumps({k: v for k, v in new_gate.items()
                                if k != "new_records"}, indent=2,
                               ensure_ascii=False), encoding="utf-8")
                info.update(d_fail=new_gate["d_fail"],
                            d_replay=new_gate["d_replay"])
                passes = new_gate["d_fail"] >= self.cfg.accept_delta
                regressed = (mode == "PATCH"
                             and new_gate["d_replay"] < -self.cfg.replay_tol)
                if passes and not regressed:
                    info.update(accepted=True, outcome="accepted")
                    return (True, revision, new_gate, rounds,
                            list(lint_report.get("llm_residual") or []))
                # A revision that wins back the fail set but breaks the replay
                # set fails its round: C1's regression retry is a repair of an
                # otherwise accepted patch, not another attempt inside a round.
                info["outcome"] = "replay_regression" if passes else "gate"
                # G5: the better `d_fail` is the base of the next round — but
                # in PATCH mode only if it did not break the replay set. A
                # revision that wins the fail items by losing what the skill
                # could already do is a dead end: every later round would be
                # drafted from a document that can no longer be accepted.
                if (new_gate["d_fail"] > base_gate["d_fail"]
                        and not (mode == "PATCH" and regressed)):
                    base_doc, base_gate = revision, new_gate
            except Exception as exc:  # noqa: BLE001 — one round, not the trigger
                # NOT `_note_error`: that counts an `errors`, and a lost extra
                # round is not a failed trigger. The round says what happened
                # in the action record, which is where the run's readers look.
                info.update(outcome="error",
                            error=f"{type(exc).__name__}: {exc}")
                sys.stderr.write(f"[warn] revision round {n} for {skill_id} "
                                 f"failed: {type(exc).__name__}: {exc} — "
                                 f"rejecting the candidate as usual\n")
                break
        return False, candidate, gate, rounds, []

    def _routing_check(self, uids: List[str], *, expect_id: str, trace: Path,
                       allow_repair: bool, diag: Dict[str, Any],
                       coherent: List[str],
                       min_hit: float = 0.6,
                       max_workers: int | None = None) -> float | None:
        """Planner-only check: does the manifest route these questions here?

        Returns None when nothing could be measured (no targets, or every
        planner call failed) — never a perfect 1.0 that hides a broken check.
        """
        targets = list(dict.fromkeys(
            uids + [e["uid"] for e in self.state.replay.get(expect_id, [])]))
        items = [self.by_uid[u] for u in targets if u in self.by_uid]
        if not items:
            return None

        def _measure() -> tuple[float | None, Dict[str, List[str]]]:
            with self._lock:      # a concurrent `_commit` must not be half-seen
                manifest = self.library.manifest()
            picks: Dict[str, List[str]] = {}
            lock = threading.Lock()

            def _one(item: StreamItem, client: Any, _wid: int) -> None:
                ids = self.route_fn(item, manifest, client)
                with lock:
                    picks[item.uid] = list(ids)

            # Every routing check used to take the whole `cfg.workers` budget,
            # and up to `trigger_workers` checks run at once — 8 × 64 planner
            # calls in flight against 16 keys. The share is passed in by the
            # caller, exactly as the distillation phase does.
            self._worker_pool(items, _one, label="planner_route",
                              max_workers=max_workers)
            if not picks:
                return None, picks   # every planner call failed
            hits = sum(1 for ids in picks.values() if expect_id in ids)
            return hits / len(picks), picks

        hit, picks = _measure()
        log: Dict[str, Any] = {
            "expect_id": expect_id,
            "hit_rate": (round(hit, 4) if hit is not None else None),
            "n_targets": len(items), "n_measured": len(picks), "picks": picks}
        if hit is not None and hit < min_hit and allow_repair:
            payload = dict(diag)
            payload["routing_note"] = (
                f"The planner routes only {hit:.0%} of these questions to "
                f"'{expect_id}'. Return REROUTE with redescribe entries that fix "
                f"the descriptions so the planner picks it from the question text.")
            with self._lock:
                manifest = self.library.manifest()
            curation, _p, raws = self.agents.curate(
                manifest=manifest, skill_in_use_id=expect_id,
                diagnosis_json=json.dumps(payload, ensure_ascii=False, indent=2),
                example_questions=self._example_questions(coherent))
            log["reroute_raw"] = raws
            if curation:
                # `_apply_redescribe` takes the library lock itself now (D6):
                # holding it here would keep it across its repair call.
                done = self._apply_redescribe(
                    curation.get("redescribe") or [], stage="routing_check")
                log["redescribed"] = done
                if done:
                    with self._lock:
                        self.state.counters["rerouted"] += 1
            hit2, picks2 = _measure()
            log["hit_rate_after"] = (round(hit2, 4) if hit2 is not None else None)
            log["n_measured_after"] = len(picks2)
            log["picks_after"] = picks2
            hit = hit2
        (trace / "routing.json").write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        shown = "n/a (nothing measured)" if hit is None else f"{hit:.2f}"
        self.log(f"    routing hit rate for {expect_id}: {shown}")
        return hit

    # ------------------------------------------- C6: cluster → merge → refine
    def _exam_set(self, members: Sequence[str]) -> List[str]:
        """Everything these skills are responsible for, capped and stratified.

        Round-robin over the members so that a skill with a big replay set
        cannot fill the whole exam; deterministic, so a resumed run examines
        the same items.
        """
        per_member: Dict[str, List[str]] = {}
        for sid in members:
            uids = [e["uid"] for e in self.state.replay.get(sid, [])]
            uids += self._recent_first(self._pool(sid))
            per_member[sid] = [u for u in dict.fromkeys(uids)
                               if u in self.by_uid]
        out: List[str] = []
        seen: set[str] = set()
        for i in range(max((len(v) for v in per_member.values()), default=0)):
            for sid in members:
                if len(out) >= self.cfg.exam_cap:
                    return out
                lst = per_member[sid]
                if i < len(lst) and lst[i] not in seen:
                    seen.add(lst[i])
                    out.append(lst[i])
        return out

    def _skill_doc(self, sid: str, library: SkillLibrary | None = None
                   ) -> str:
        skill = (library or self.library).get(sid)
        if skill is None:
            return "(none)"
        return (f"---\nid: {skill.id}\ndescription: {skill.description}\n---"
                f"\n\n{skill.body}")

    def _render_skills_for_cluster(self, ids: Sequence[str],
                                   library: SkillLibrary | None = None) -> str:
        """id, description and the "When to use" section of every skill."""
        lib = library or self.library
        blocks = []
        for sid in ids:
            skill = lib.get(sid)
            if skill is None:
                continue
            when = _section(skill.body, "When to use")
            blocks.append(f"### {sid}\ndescription: {skill.description}\n"
                          f"when to use: {when or '(not stated)'}")
        return "\n\n".join(blocks) if blocks else "(none)"

    def _render_exam(self, uids: Sequence[str],
                     results: Dict[str, Dict[str, Any]],
                     others: Dict[str, Dict[str, Dict[str, Any]]] | None = None
                     ) -> str:
        """The exam as the refiner sees it: every item, complete expected."""
        blocks: List[str] = []
        for uid in uids:
            res = results.get(uid)
            base = self.state.records.get(uid, {})
            if res is None:
                continue
            parts = [
                f"### case {uid}",
                f"eval_metric: {res.get('eval_metric', base.get('eval_metric', ''))}",
                f"question: {scrub(res.get('question', base.get('question', '')))}",
                f"expected: {scrub(res.get('ground_truth', base.get('ground_truth', '')))[:self.cfg.gt_max_chars]}",
                f"produced: {scrub(res.get('prediction', ''))[:600]}",
                f"score: {float(res.get('score', 0.0)):.3f}",
                f"executor: {res.get('exec_status', '')} — "
                f"{str(res.get('exec_message', ''))[:400] or '(no message)'}",
                "final code:\n```python\n"
                + scrub(truncate_code(str(res.get("code") or ""),
                                      self.cfg.code_max_lines)) + "\n```",
            ]
            if float(res.get("score", 0.0)) < self.cfg.pass_thr and others:
                for other_id, other_res in sorted(others.items()):
                    r = other_res.get(uid)
                    if r is None or float(r.get("score", 0.0)) < self.cfg.pass_thr:
                        continue
                    parts.append(
                        f"ANOTHER SKILL ({other_id}) answers this one "
                        f"(score {float(r['score']):.3f}) with:\n```python\n"
                        + scrub(truncate_code(str(r.get("code") or ""),
                                              self.cfg.code_max_lines))
                        + "\n```")
                    break
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks) if blocks else "(none)"

    def _age_index(self, sid: str) -> int:
        """How old a skill is: its position in the checkpoint's version map."""
        order = list(self.state.skill_versions.keys())
        return order.index(sid) if sid in order else len(order)

    def _refiner_doc(self, *, mode: str, skill_id: str, description: str,
                     eval_metrics: str, primary_skill: str, other_skills: str,
                     schema_example: str, exam_results: str,
                     reference_solutions: str) -> str | None:
        """One refiner call plus the C2 retry, normalised like a draft."""
        kw = dict(mode=mode, skill_id=skill_id, description=description,
                  eval_metrics=eval_metrics, primary_skill=primary_skill,
                  other_skills=other_skills, schema_example=schema_example,
                  exam_results=exam_results,
                  reference_solutions=reference_solutions)
        raw, _p, finish = self.agents.refine(**kw)
        truncated = finish == "length"
        doc = None if truncated else normalize_doc(raw, skill_id, description)
        if doc is not None:
            return doc
        raw2, _p, finish2 = self.agents.refine(
            previous_reply=raw,
            reminder=(LENGTH_REMINDER if truncated else DOC_REMINDER), **kw)
        if finish2 == "length":
            return None
        return normalize_doc(raw2, skill_id, description)

    def _consolidation_lint(self, doc: str, *, skill_id: str,
                            library: SkillLibrary,
                            questions: Sequence[str]) -> tuple[str | None,
                                                               Dict[str, Any]]:
        """Mechanical lint with one repair pass, as a hard gate (C6)."""
        report: Dict[str, Any] = {"passes": []}
        text = doc
        for attempt in (1, 2):
            mech = self.mech_lint(text, skill_id=skill_id, questions=questions,
                                  library=library)
            report["passes"].append({"attempt": attempt, "mechanical": mech})
            if mech["verdict"] == "pass":
                report["verdict"] = "pass"
                return text, report
            if attempt == 2:
                break
            bullets = self._repair_bullets(mech["violations"])
            repaired = self._draft_doc(
                mode="PATCH", skill_id=skill_id,
                description=(parse_skill_text(text).description
                             if parse_skill_text(text) else ""),
                eval_metrics="(unknown)", question_intent="",
                error_patterns=(self._repair_header(mech["violations"])
                                + "\n" + bullets),
                success_notes="(none)", current_skill=text,
                example_questions="(none)")
            if repaired is None:
                break
            text = repaired
        report["verdict"] = "fail"
        return None, report

    def consolidate(self, *, refine: bool, reason: str) -> Dict[str, Any]:
        """Cluster the library, merge duplicates, then refine (ADDENDUM C6)."""
        t0 = time.time()
        self.state.counters["consolidations"] += 1
        n = int(self.state.counters["consolidations"])
        trace = self.traces_dir / f"consolidate_{n:02d}"
        trace.mkdir(parents=True, exist_ok=True)
        ids = self.library.ids()
        record: Dict[str, Any] = {
            "consolidate": n, "reason": reason, "refine": refine,
            "n_skills_before": len(ids), "clusters": [], "merges": [],
            "retired": [], "refines": [], "routing": []}
        self.log(f"[consolidate {n}] {reason}: {len(ids)} skills")
        if len(ids) < 1:
            return self._log_consolidation(record, t0)

        snapshot = self._snapshot_library()
        try:
            lib = SkillLibrary(snapshot.name)
            try:
                parsed, _p, raws = self.agents.cluster(
                    skills=self._render_skills_for_cluster(ids, lib))
            except Exception as exc:  # noqa: BLE001 — all singletons, no merge
                self._note_error("cluster call", exc)
                parsed, raws = None, [f"{type(exc).__name__}: {exc}"]
            clusters = assign_clusters(parsed, ids)
            (trace / "cluster_out.json").write_text(
                json.dumps({"parsed": parsed, "raw": raws,
                            "clusters": [{"shared_intent": i, "skill_ids": m}
                                         for i, m in clusters]},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            record["clusters"] = [{"shared_intent": i, "skill_ids": m}
                                  for i, m in clusters]
            self.log(f"[consolidate {n}] clusters: "
                     + "; ".join("+".join(m) for _i, m in clusters))

            multi = [c for c in clusters if len(c[1]) >= 2]
            share = max(1, self.cfg.workers // max(1, len(multi) or 1))

            def _one_merge(c: Tuple[str, List[str]]) -> Dict[str, Any]:
                # One cluster's plan, like one trigger's: a failed LLM call
                # must cost that cluster, not the whole consolidation.
                try:
                    return self._plan_merge(c, library=lib, trace=trace,
                                            max_workers=share)
                except Exception as exc:  # noqa: BLE001
                    self._note_error(f"merge plan for {'+'.join(c[1])}", exc)
                    return {"cluster": list(c[1]), "shared_intent": c[0],
                            "winner": "", "merged_accepted": False,
                            "retire": [], "keep": [],
                            "error": f"{type(exc).__name__}: {exc}"}

            plans = _map_parallel(
                _one_merge, multi,
                max(1, min(self.cfg.trigger_workers, len(multi) or 1)),
                name="cs-merge") if multi else []
            for plan in sorted(plans, key=lambda p: str(p.get("winner", ""))):
                self._apply_merge(plan, record=record, trace=trace)
            self.save_state()
        finally:
            snapshot.cleanup()

        if refine:
            self._refine_survivors(record=record, trace=trace)
        self._consolidation_routing(record=record, trace=trace)
        record["n_skills_after"] = len(self.library.ids())
        self.save_state()
        return self._log_consolidation(record, t0)

    def _log_consolidation(self, record: Dict[str, Any],
                           t0: float) -> Dict[str, Any]:
        record.setdefault("n_skills_after", len(self.library.ids()))
        record["seconds"] = round(time.time() - t0, 1)
        record["llm_calls"] = self._llm_base + self.stats.calls
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.log(f"[consolidate {record['consolidate']}] "
                 f"{record['n_skills_before']} → {record['n_skills_after']} "
                 f"skills, {len(record['retired'])} retired")
        return record

    def _plan_merge(self, cluster: Tuple[str, List[str]], *,
                    library: SkillLibrary, trace: Path,
                    max_workers: int | None) -> Dict[str, Any]:
        """Exam the cluster, write one merged skill, gate it — no writes."""
        intent, members = cluster
        # Short enough for a file name however big the cluster is.
        label = f"{members[0]}+{len(members) - 1}"
        plan: Dict[str, Any] = {"cluster": members, "shared_intent": intent,
                                "winner": "", "merged_accepted": False,
                                "retire": [], "keep": []}
        exam = self._exam_set(members)
        plan["exam"] = exam
        if not exam:
            plan["skipped"] = "no_exam_items"
            return plan
        items = [self.by_uid[u] for u in exam]
        share = max(1, (max_workers or self.cfg.workers) // len(members))
        per_member = {
            sid: self._solve(items, forced_skill_ids=[sid], library=library,
                             max_workers=share) for sid in members}
        means = {sid: mean([r["score"] for r in res.values()])
                 for sid, res in per_member.items()}
        plan["member_means"] = {k: round(v, 4) for k, v in means.items()}
        winner = sorted(members,
                        key=lambda s: (-means[s],
                                       -len(self.state.replay.get(s, [])),
                                       self._age_index(s), s))[0]
        plan["winner"] = winner
        others = [s for s in members if s != winner]

        # Every member's replay items that the exam cap left out. The exam is
        # what the MERGE gate and the substitutability rule are decided on, so
        # without these two decisions are taken on whatever slice of a
        # member's practice happened to fit: a member can be retired on one of
        # its twelve items, and a merged document can be committed although it
        # destroys replay items of the winner it replaces.
        in_exam = set(exam)
        extra = [u for u in dict.fromkeys(
            e["uid"] for sid in members for e in self.state.replay.get(sid, []))
            if u not in in_exam and u in self.by_uid]
        extra_items = [self.by_uid[u] for u in extra]
        plan["n_extra_replay"] = len(extra)
        winner_extra = (self._solve(extra_items, forced_skill_ids=[winner],
                                    library=library, max_workers=max_workers)
                        if extra_items else {})

        doc = self._refiner_doc(
            mode="MERGE", skill_id=winner,
            description=(library.get(winner).description
                         if library.get(winner) else ""),
            eval_metrics=self._eval_metrics(exam),
            primary_skill=self._skill_doc(winner, library),
            other_skills="\n\n".join(self._skill_doc(s, library)
                                     for s in others) or "(none)",
            schema_example=self._schema_example(exam),
            exam_results=self._render_exam(
                exam, per_member[winner],
                {s: per_member[s] for s in others}),
            reference_solutions=self._reference_solutions(winner, exam))
        (trace / f"merge_{label}_draft.md").write_text(doc or "(unparsable)",
                                                       encoding="utf-8")
        merged_scores: Dict[str, float] = {}
        if doc is not None:
            doc, lint_report = self._consolidation_lint(
                doc, skill_id=winner, library=library,
                questions=self._questions_of(exam))
            (trace / f"merge_{label}_lint.json").write_text(
                json.dumps(lint_report, indent=2, ensure_ascii=False),
                encoding="utf-8")
        if doc is not None:
            tmp = self._temp_library(winner, doc, library.dir)
            try:
                merged_lib = SkillLibrary(tmp.name)
                new = self._solve(items, forced_skill_ids=[winner],
                                  library=merged_lib, max_workers=max_workers)
                merged_mean = mean([r["score"] for r in new.values()])
                merged_extra = (
                    self._solve(extra_items, forced_skill_ids=[winner],
                                library=merged_lib, max_workers=max_workers)
                    if extra_items and merged_mean >= means[winner] else {})
            finally:
                tmp.cleanup()
            candidate = {u: float(r["score"])
                         for u, r in list(new.items()) + list(merged_extra.items())}
            # A replay guard for MERGE, mirroring `_gate`'s `d_replay` for
            # PATCH: the merged document replaces the winner in place, so it
            # may not lose the winner's own practice.
            d_replay = self._replay_delta(winner, candidate)
            plan.update(merged_mean=round(merged_mean, 4),
                        winner_mean=round(means[winner], 4),
                        merged_d_replay=round(d_replay, 4))
            if merged_mean >= means[winner] and d_replay >= -self.cfg.replay_tol:
                plan["merged_accepted"] = True
                plan["doc"] = doc
                merged_scores = candidate
            elif merged_mean >= means[winner]:
                plan["merge_rejected"] = "replay_regression"
            (trace / f"merge_{label}_gate.json").write_text(
                json.dumps({"winner": winner, "winner_mean": means[winner],
                            "merged_mean": merged_mean,
                            "d_replay": d_replay,
                            "n_extra_replay": len(extra),
                            "accepted": plan["merged_accepted"]},
                           indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            plan["merge_failed"] = True

        # Who can the survivor replace? Measured on each member's OWN replay
        # items — ALL of them, not only those the exam cap happened to keep —
        # with the scores those items were recorded with. A member that never
        # practised (C6 was written for exactly those unpractised
        # near-duplicates) is judged on its open failures instead.
        survivor = dict(merged_scores) if plan["merged_accepted"] else {
            u: float(r["score"]) for u, r in
            list(per_member[winner].items()) + list(winner_extra.items())}
        for sid in others:
            own, evidence = self._substitution_evidence(sid, survivor)
            if not own:
                plan["keep"].append({"skill_id": sid, "why": "no_evidence"})
                continue
            own_mean = mean([s for _u, s in own])
            surv_mean = mean([survivor[u] for u, _s in own])
            entry = {"skill_id": sid, "own_mean": round(own_mean, 4),
                     "survivor_mean": round(surv_mean, 4), "n": len(own),
                     "evidence": evidence}
            if surv_mean >= own_mean - self.cfg.replay_tol:
                plan["retire"].append(entry)
            else:
                plan["keep"].append({**entry, "why": "not_substitutable"})
        return plan

    def _replay_delta(self, sid: str, scores: Dict[str, float]) -> float:
        """`mean(new) − mean(recorded)` over the replay items we could measure."""
        pairs = [(float(e["score"]), scores[e["uid"]])
                 for e in self.state.replay.get(sid, []) if e["uid"] in scores]
        if not pairs:
            return 0.0
        return mean([n for _o, n in pairs]) - mean([o for o, _n in pairs])

    def _substitution_evidence(self, sid: str, survivor: Dict[str, float]
                               ) -> Tuple[List[Tuple[str, float]], str]:
        """`([(uid, own score)], what the evidence is)` for retiring `sid`.

        The replay set is the evidence when there is one. A forked
        near-duplicate typically has none — SPEC C6.3's criterion is then
        vacuously true, but the implementation used to read that as "keep",
        which disarms the budget mechanism for precisely the population it was
        written for. Its open failures, which the survivor was just run on,
        are the next best evidence.
        """
        own = [(e["uid"], float(e["score"]))
               for e in self.state.replay.get(sid, []) if e["uid"] in survivor]
        if own:
            return own, "replay"
        pool = [(u, float(self.state.records.get(u, {}).get("score", 0.0)))
                for u in self._pool(sid) if u in survivor]
        return pool, "fail_pool"

    def _apply_merge(self, plan: Dict[str, Any], *, record: Dict[str, Any],
                     trace: Path) -> None:
        winner = str(plan.get("winner") or "")
        if not winner or self.library.get(winner) is None:
            record["merges"].append({**{k: v for k, v in plan.items()
                                        if k != "doc"},
                                     "applied": False})
            return
        if plan.get("merged_accepted") and plan.get("doc"):
            self._commit(winner, str(plan["doc"]))
            self.state.counters["merged"] += 1
        for entry in plan.get("retire", []):
            moved = self._retire_skill(str(entry["skill_id"]), survivor=winner)
            record["retired"].append({**entry, "survivor": winner, **moved})
        for entry in plan.get("keep", []):
            sid = str(entry["skill_id"])
            # Only a member that really is NOT substitutable needs a sharper,
            # more distinct description; asking the curator to sharpen one
            # that was merely unmeasurable makes it *more* attractive to the
            # planner for no reason.
            if entry.get("why") != "not_substitutable":
                continue
            if self.library.get(sid) is None:
                continue
            try:
                self._separate_descriptions(sid, winner, trace=trace)
            except Exception as exc:  # noqa: BLE001
                self._note_error(f"description separation for {sid}", exc)
        record["merges"].append({**{k: v for k, v in plan.items()
                                    if k != "doc"}, "applied": True})

    def _retire_skill(self, sid: str, *, survivor: str) -> Dict[str, int]:
        """Move a superseded skill out of the library, keeping its evidence."""
        src = self.lib_dir / f"{sid}.md"
        dst = self.history_dir / RETIRED_DIR / sid
        dst.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            version = int(self.state.skill_versions.get(sid, 0))
            src.replace(dst / f"v{version or 1}.md")
        # The whole history goes with it, and the version counter is dropped:
        # otherwise a later skill that happens to be minted under the same id
        # starts at the retired skill's version, writes into its history and
        # is listed in `state.retired` although it is alive.
        hist = self.history_dir / sid
        if hist.is_dir():
            for p in sorted(hist.iterdir()):
                target_path = dst / p.name
                if target_path.exists():
                    target_path.unlink()
                p.replace(target_path)
            with suppress(OSError):
                hist.rmdir()
        self.state.skill_versions.pop(sid, None)
        # Its practice and its open failures belong to the survivor now.
        moved = self._merge_replay(survivor, self.state.replay.pop(sid, []))
        pool = self.state.fail_pool.pop(sid, {})
        target = self._pool(survivor)
        for uid, attempts in pool.items():
            target[uid] = max(target.get(uid, 0), attempts)
        self.state.replay_seen.pop(sid, None)
        self.state.retired[sid] = survivor
        self.state.counters["retired"] += 1
        self.log(f"    retired {sid} → {survivor} "
                 f"({moved['replay_kept']}/{moved['replay_transferred']} "
                 f"replay items transferred)")
        return moved

    def _merge_replay(self, survivor: str,
                      entries: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """Absorb a retired skill's replay set into the survivor's, directly.

        Pushing the entries through `_add_replay` would run them through the
        reservoir: a survivor that already holds `replay_cap` items keeps a
        transferred item only with probability `cap/(seen+1)`, and every item
        it does keep evicts one of its own. On the real full_v1 counters that
        threw away 10 of 12 transferred items and 2 of the survivor's own —
        and the replay set is the ONLY regression guard the C1 gate has, so
        every lost item is a capability a later patch can destroy unnoticed.
        Here the two sets are interleaved instead, so a cap (twice the normal
        one, because the survivor really is responsible for two skills' work
        now) falls on both sides evenly and deterministically.
        """
        entries = [dict(e) for e in entries]
        with self._lock:
            own = self.state.replay.setdefault(survivor, [])
            by_uid = {e["uid"]: float(e["score"]) for e in own}
            incoming = [e for e in entries if e["uid"] not in by_uid]
            merged: List[Dict[str, Any]] = []
            for i in range(max(len(own), len(incoming))):
                if i < len(own):
                    merged.append(own[i])
                if i < len(incoming):
                    merged.append(incoming[i])
            merged = merged[:2 * self.cfg.replay_cap]
            self.state.replay[survivor] = merged
            kept = sum(1 for e in merged if e["uid"] in
                       {x["uid"] for x in entries})
            self.state.replay_seen[survivor] = max(
                int(self.state.replay_seen.get(survivor, 0)), len(merged))
        return {"replay_transferred": len(entries), "replay_kept": kept}

    def _separate_descriptions(self, sid: str, other: str, *,
                               trace: Path) -> None:
        """A member that survives a merge must not overlap the survivor (C6)."""
        payload = {
            "reroute_note": (
                f"'{sid}' and '{other}' both stay in the library, but their "
                f"descriptions overlap and the planner cannot tell them apart. "
                f"Return REROUTE with a redescribe entry for BOTH ids so that "
                f"each description says what it is for and what it is not for."),
        }
        curation, _p, raws = self.agents.curate(
            manifest=self.library.manifest(), skill_in_use_id=sid,
            diagnosis_json=json.dumps(payload, ensure_ascii=False, indent=2),
            example_questions=self._example_questions(
                self._recent_first(self._pool(sid))))
        (trace / f"separate_{sid}.json").write_text(
            json.dumps({"parsed": curation, "raw": raws}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        if curation:
            if self._apply_redescribe(curation.get("redescribe") or [],
                                      stage="consolidate_separate"):
                self.state.counters["rerouted"] += 1

    def _refine_survivors(self, *, record: Dict[str, Any],
                          trace: Path) -> None:
        """`refine_rounds` REFINE rounds over the surviving skills (C6)."""
        for rnd in range(max(0, self.cfg.refine_rounds)):
            ids = [sid for sid in self.library.ids()
                   if len(self._exam_set([sid])) >= self.cfg.final_min_fail]
            if not ids:
                return
            sub = trace / f"refine_{rnd + 1}"
            sub.mkdir(parents=True, exist_ok=True)
            snapshot = self._snapshot_library()
            try:
                lib = SkillLibrary(snapshot.name)
                share = max(1, self.cfg.workers // len(ids))

                def _one_refine(sid: str) -> Dict[str, Any]:
                    try:
                        return self._plan_refine(sid, library=lib, trace=sub,
                                                 max_workers=share)
                    except Exception as exc:  # noqa: BLE001
                        self._note_error(f"refine plan for {sid}", exc)
                        return {"skill_id": sid, "accepted": False,
                                "skipped": f"{type(exc).__name__}: {exc}"}

                plans = _map_parallel(
                    _one_refine, ids,
                    max(1, min(self.cfg.trigger_workers, len(ids))),
                    name="cs-refine")
            finally:
                snapshot.cleanup()
            for plan in sorted(plans, key=lambda p: str(p["skill_id"])):
                if plan.get("accepted") and plan.get("doc"):
                    self._commit(str(plan["skill_id"]), str(plan["doc"]))
                    self.state.counters["refined"] += 1
                record["refines"].append({**{k: v for k, v in plan.items()
                                             if k != "doc"},
                                          "round": rnd + 1})
            self.save_state()

    def _plan_refine(self, sid: str, *, library: SkillLibrary, trace: Path,
                     max_workers: int | None) -> Dict[str, Any]:
        plan: Dict[str, Any] = {"skill_id": sid, "accepted": False}
        exam = self._exam_set([sid])
        items = [self.by_uid[u] for u in exam]
        plan["n_exam"] = len(items)
        if len(items) < self.cfg.final_min_fail:
            plan["skipped"] = "exam_too_small"
            return plan
        old = self._solve(items, forced_skill_ids=[sid], library=library,
                          max_workers=max_workers)
        old_mean = mean([r["score"] for r in old.values()])
        plan["old_mean"] = round(old_mean, 4)
        doc = self._refiner_doc(
            mode="REFINE", skill_id=sid,
            description=(library.get(sid).description
                         if library.get(sid) else ""),
            eval_metrics=self._eval_metrics(exam),
            primary_skill=self._skill_doc(sid, library),
            other_skills="(none)",
            schema_example=self._schema_example(exam),
            exam_results=self._render_exam(exam, old),
            reference_solutions=self._reference_solutions(sid, exam))
        if doc is None:
            plan["skipped"] = "draft_unparsable"
            return plan
        doc, lint_report = self._consolidation_lint(
            doc, skill_id=sid, library=library,
            questions=self._questions_of(exam))
        (trace / f"refine_{sid}_lint.json").write_text(
            json.dumps(lint_report, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if doc is None:
            plan["skipped"] = "lint_failed"
            return plan
        (trace / f"refine_{sid}_draft.md").write_text(doc, encoding="utf-8")
        tmp = self._temp_library(sid, doc, library.dir)
        try:
            new = self._solve(items, forced_skill_ids=[sid],
                              library=SkillLibrary(tmp.name),
                              max_workers=max_workers)
        finally:
            tmp.cleanup()
        new_mean = mean([r["score"] for r in new.values()])
        plan["new_mean"] = round(new_mean, 4)
        plan["accepted"] = new_mean >= old_mean + self.cfg.refine_delta
        if plan["accepted"]:
            plan["doc"] = doc
        (trace / f"refine_{sid}_gate.json").write_text(
            json.dumps({k: v for k, v in plan.items() if k != "doc"},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        return plan

    def _consolidation_routing(self, *, record: Dict[str, Any],
                               trace: Path) -> None:
        """Step 5: does the final manifest route the exam items right?"""
        ids = self.library.ids()
        if not ids:
            return
        jobs = [(sid, self._exam_set([sid])) for sid in ids]
        jobs = [(sid, uids) for sid, uids in jobs if uids]
        if not jobs:
            return

        share = max(1, self.cfg.workers // len(jobs))

        def _one(job: Tuple[str, List[str]]) -> Dict[str, Any]:
            sid, uids = job
            sub = trace / f"routing_{sid}"
            sub.mkdir(parents=True, exist_ok=True)
            try:
                hit = self._routing_check(uids, expect_id=sid, trace=sub,
                                          allow_repair=True, diag={},
                                          coherent=uids,
                                          min_hit=self.cfg.routing_min_hit,
                                          max_workers=share)
            except Exception as exc:  # noqa: BLE001 — a report, not a decision
                self._note_error(f"consolidation routing for {sid}", exc)
                return {"skill_id": sid, "hit_rate": None, "n": len(uids),
                        "error": f"{type(exc).__name__}: {exc}"}
            return {"skill_id": sid, "hit_rate": hit, "n": len(uids)}

        results = _map_parallel(_one, jobs,
                                max(1, min(self.cfg.trigger_workers,
                                           len(jobs))), name="cs-croute")
        record["routing"] = list(results)

    # ------------------------------------------------------------- guards
    # Runtime guard: ONLY strings that cannot come from anywhere but the
    # benchmark labels. Ordinary English the LLM may legitimately produce
    # ("category", "subtask", "sliding window", …) is checked in
    # tests/test_label_blind.py against real items, never at runtime — a
    # false positive there would throw away a trigger for nothing.
    _TASK_ID_RE = re.compile(r"\bL[1-4]_[A-Za-z0-9_]+")

    def _assert_label_blind(self, prompt: str) -> None:
        """Cheap runtime guard: no benchmark task id may reach a prompt."""
        m = self._TASK_ID_RE.search(prompt)
        if m is not None:
            raise LabelLeak(f"label leak: prompt contains a task id "
                            f"({m.group(0)!r})")
        low = prompt.lower()
        for tid in self._task_id_tokens:
            if tid and tid in low:
                raise LabelLeak(f"label leak: prompt contains a task id "
                                f"({tid!r})")

    # ------------------------------------------------------------- summary
    def write_summary(self, *, seconds: float) -> Dict[str, Any]:
        final_dir = self.out / "library_final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        for md in self.lib_dir.glob("*.md"):
            shutil.copy2(md, final_dir / md.name)

        # C2/C7: state plainly whether every shipped skill is mechanically clean.
        failing = [r["skill_id"] for r in self.lint_library_report()]
        summary = {
            "library_dir": str(self.lib_dir),
            "manifest": self.library.manifest(),
            "mech_lint_failing": failing,
            "skills": [
                {"id": sid,
                 "version": self.state.skill_versions.get(sid, 0),
                 "replay": len(self.state.replay.get(sid, [])),
                 "description": (self.library.get(sid).description
                                 if self.library.get(sid) else "")}
                for sid in self.library.ids()
            ],
            "retired": dict(self.state.retired),
            "counters": self.state.counters,
            # D1/D7: what the three conditions have to be comparable on —
            # the LLM worker count, the execution concurrency the subprocess
            # semaphore was built with, and how the runs actually ended.
            "workers": self.cfg.workers,
            "trigger_workers": self.cfg.trigger_workers,
            "exec_concurrency": EXEC_CONCURRENCY,
            # `exec_cfg` is the dict `run.py` builds out of online.yaml.
            "exec_timeout_seconds": (self.exec_cfg or {}).get("timeout_seconds"),
            "exec_status_counts": dict(
                self.state.counters.get("exec_status_counts") or {}),
            "pipeline_runs": int(self.state.counters.get("pipeline_runs", 0)),
            "redescribe_rejected_by_rule": dict(
                self.state.counters.get("redescribe_rejected_by_rule") or {}),
            "llm": self.stats.snapshot(),
            "items_in_stream": len(self.sequence),
            "pool_sizes": {k: len(v) for k, v in self.state.fail_pool.items() if v},
            "mean_score_last_seen": round(
                mean([r["score"] for r in self.state.records.values()]), 4),
            "seconds": round(seconds, 1),
        }
        (self.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return summary
