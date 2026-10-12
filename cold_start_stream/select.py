"""CLI: pick, on held-out VALID items, which distilled skills to keep (E3).

    python -m cold_start_stream.select --library <dir> [--library <dir> ...]
        --tasks <tasks.json> --databases <copy of the VALID databases>
        --ids-file <valid ids json> --out <select dir> [--workers W]

Three independent streams produce three libraries of the same high-variance
search. Nothing in the streaming procedure ever measured a skill on data it was
not distilled from: the gate compares a candidate only on the items that had
FAILED, whose old scores are low by selection. This module is the unbiased
half: pool the three libraries, cluster the near-duplicates, and for every
cluster that enough VALID questions route to, keep the single best member —
and only if it beats an empty library by `--select-margin` on those items.
Finally, under real planner routing, drop any kept skill that turns out to
harm the items it actually attracts.

Everything here uses labels only in the sense that VALID is held out: the
decisions read scores, not `subtask` / `category` / `level`.


What one selection costs (ADDENDUM H9)
--------------------------------------
Round 2 spent **6 338 LLM calls (≈45 RMB)** on one `select` command — about
80 % of the whole round — because nothing bounded the middle term below. The
number is not a mystery, it is arithmetic; write it out before launching:

    pipeline_runs = repeats * ( |V|                       # NONE reference
                              + |V|                       # do-no-harm, 1 round
                              + Σ_c |M_c| · min(|V_c|, max_items_per_cluster) )
    llm_calls     ≈ |V|                                   # planner routing
                  + k · pipeline_runs                     # k ≈ 2: plan + code,
                                                          # more on regeneration

`|V|` = `--ids-file` items, `M_c` = members of cluster c, `V_c` = the VALID
items the planner routes to any member of c, `repeats` = `--repeats`.
A second do-no-harm round (only when round 1 drops someone) adds `repeats·|V|`.

Round 2 measured: |V| = 300, repeats = 2, 11 skills in 5 clusters with
|V_c| = 150 / 53 / 24 / 55 / 36 and |M_c| = 4 / 2 / 2 / 2 / 1, no second
do-no-harm round →
    Σ_c |M_c|·|V_c| = 600 + 106 + 48 + 110 + 36 = 900
    pipeline_runs   = 2·(300 + 300 + 900) = 3 000
    llm_calls       ≈ 300 + 2·3 000 = 6 300  (measured 6 338)

`--max-items-per-cluster` (default 120) caps only the middle term: a cluster
whose `V_c` is larger is validated on a seeded random subset of that size.
With the default, round 2's cluster 1 would have run on 120 instead of 150
items — Σ drops from 900 to 780, i.e. 240 fewer pipeline runs (≈480 calls).
The NONE reference and the do-no-harm stage are deliberately NOT subsampled:
they are the two full-`|V|` numbers the report compares everything against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Sequence,
                    Tuple)

import yaml

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from sonar_ts import _paths as paths_mod  # noqa: E402
from sonar_ts import planner as plan_mod  # noqa: E402
from sonar_ts.evaluator import score_one  # noqa: E402
from sonar_ts.executor import EXEC_CONCURRENCY  # noqa: E402
from sonar_ts.llm import DeepSeekClient, load_api_keys  # noqa: E402
from sonar_ts.pipeline import run_task as _default_run_task  # noqa: E402
from sonar_ts.schema import build_schema_text  # noqa: E402
from sonar_ts.skills import SkillLibrary, parse_skill_text  # noqa: E402

from .agents import Agents, CallStats, CountingClient, PROMPTS_DIR  # noqa: E402
from .guards import (check_databases_dir,  # noqa: E402
                     check_symbolization_env)
from .index_usage import index_usage  # noqa: E402
from .lint import (DEFAULT_DOC_MAX_CHARS, DEFAULT_SNIPPET_MAX_LINES,  # noqa: E402
                   mechanical_lint)
from .orchestrator import _section, assign_clusters, mean  # noqa: E402

_CONFIG_PATH = _FRAMEWORK_ROOT / "configs" / "online.yaml"


# --------------------------------------------------------------------------- config
@dataclass
class SelectConfig:
    """The knobs of the selection (all overridable from the CLI)."""

    repeats: int = 2                 # E3: every measurement is run twice
    select_margin: float = 0.03      # E3.5: keep BEST only if it is this much better
    min_valid_items: int = 5         # E3.5: |V_c| below this is not validated
    harm_margin: float = 0.02        # E3.7: drop below −this
    harm_min_items: int = 5          # E3.7: …on at least this many items
    # H9: the cost cap. A cluster whose |V_c| exceeds this is validated on a
    # seeded random subset of exactly this many of its routed items; 0 (or a
    # negative value) means "no cap", i.e. the pre-H9 behaviour.
    max_items_per_cluster: int = 120
    seed: int = 20260922             # H9: the subsampling seed
    workers: int = 16
    doc_max_chars: int = DEFAULT_DOC_MAX_CHARS
    snippet_max_lines: int = DEFAULT_SNIPPET_MAX_LINES
    distill_max_tokens: int = 8192   # C2: the cluster call is a distillation call
    max_error_rate: float = 0.05     # a stage with more internal failures is refused

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- pool
def rewrite_skill_id(text: str, new_id: str) -> str:
    """The same document under a new frontmatter id (E3.1 clash renaming)."""
    skill = parse_skill_text(text)
    if skill is None:
        raise ValueError("not a skill document")
    lines = text.splitlines()
    out: List[str] = []
    done = False
    inside = False
    for line in lines:
        if line.strip() == "---":
            inside = not inside if not done else inside
            out.append(line)
            continue
        if inside and not done and line.strip().lower().startswith("id:"):
            out.append(f"id: {new_id}")
            done = True
            continue
        out.append(line)
    if not done:                       # no id line found: rebuild the header
        return (f"---\nid: {new_id}\ndescription: {skill.description}\n---\n\n"
                f"{skill.body}\n")
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def library_tag(index: int) -> str:
    """The suffix a clash gets: the second library's skills become `…-s2`."""
    return f"s{index}"


def condition_tag(skills: Iterable[str] | Mapping[str, str]) -> str:
    """A short, stable name for "the library that holds exactly these skills".

    The run cache is keyed by the stage name, so the name must change when the
    library does — otherwise a selection re-run with a different margin would
    read back scores that were measured under a different condition. A
    *mapping* id → document hashes the DOCUMENT BYTES as well: a stream that is
    redone or resumed can legitimately ship a different text under the same id,
    and scores measured on the old text say nothing about the new one.
    """
    if isinstance(skills, Mapping):
        if not skills:
            return "empty"
        payload = "\n".join(
            f"{sid}\0{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
            for sid, text in sorted(skills.items()))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    ids = ",".join(sorted(str(s) for s in skills))
    return hashlib.sha1(ids.encode("utf-8")).hexdigest()[:8] if ids else "empty"


def expand_all(picks: Iterable[str], library_ids: Iterable[str] | None) -> List[str]:
    """`["ALL"]` is a legal planner answer meaning "load every skill".

    `sonar_ts/prompts/task_planner.txt` documents it and `_loader._bodies_of`
    honours it, so an item whose plan says ALL really does run with every
    skill loaded. Attribution and `V_c` must see those ids, not the token.
    """
    ids = [str(p) for p in picks]
    if "ALL" not in ids:
        return ids
    rest = [i for i in ids if i != "ALL"]
    out = list(library_ids or [])
    out += [i for i in rest if i not in out]
    return out


def library_docs(library: Any) -> Dict[str, str]:
    """id → the text a `SkillLibrary` would hand the generator for that id."""
    docs: Dict[str, str] = {}
    for sid in library.ids():
        skill = library.get(sid)
        if skill is not None:
            docs[sid] = f"{skill.description}\n\n{skill.body}"
    return docs


class StageError(RuntimeError):
    """A stage whose runs failed too often to base a decision on."""


def channel_literals_of(tasks: Iterable[Dict[str, Any]]) -> set[str]:
    return {str(t.get("channel", "")).strip() for t in tasks
            if str(t.get("channel", "")).strip()}


# --------------------------------------------------------------------------- selector
class Selector:
    """Pool → cluster → route → validate → keep → do-no-harm (ADDENDUM E3)."""

    def __init__(self, *,
                 libraries: Sequence[Path | str],
                 tasks: Sequence[Dict[str, Any]],
                 out: Path | str,
                 exec_cfg: Dict[str, Any],
                 client_factory: Callable[[int], Any],
                 cfg: SelectConfig | None = None,
                 run_task_fn: Callable[..., Any] = _default_run_task,
                 route_fn: Callable[..., List[str]] | None = None,
                 prompts_dir: Path | str = PROMPTS_DIR,
                 channel_literals: Iterable[str] | None = None,
                 log: Callable[[str], None] | None = None):
        self.libraries = [Path(p) for p in libraries]
        self.tasks = list(tasks)
        self.by_id = {str(t["id"]): t for t in self.tasks}
        self.out = Path(out)
        self.exec_cfg = exec_cfg
        self.client_factory = client_factory
        self.cfg = cfg or SelectConfig()
        self.run_task_fn = run_task_fn
        self.route_fn = route_fn or _default_route
        self.prompts_dir = Path(prompts_dir)
        self.channel_literals = set(
            channel_literals_of(self.tasks) if channel_literals is None
            else channel_literals)
        self.log = log or (lambda msg: print(msg, flush=True))

        self.out.mkdir(parents=True, exist_ok=True)
        self.pool_dir = self.out / "pool"
        self.selected_dir = self.out / "library_selected"
        self.empty_dir = self.out / "_empty_library"
        self.cache_path = self.out / "runs.jsonl"
        self.routes_path = self.out / "routes.json"
        for d in (self.pool_dir, self.empty_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.stats = CallStats()
        self._lock = threading.Lock()
        self._cache = self._load_cache()
        self.pool: List[Dict[str, Any]] = []
        self.dropped: List[Dict[str, Any]] = []
        self.libraries_summary: List[Dict[str, Any]] = []
        self.stage_status: Dict[str, Dict[str, int]] = {}
        self.route_failures: List[str] = []

    # ------------------------------------------------------------ pipeline runs
    def _load_cache(self) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
        """Scores already measured, so a killed selection can be re-run.

        Not asked for by E3; the stage runs are hours of CPU and the shell that
        starts them is not allowed to block for hours.
        """
        cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        if not self.cache_path.is_file():
            return cache
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cache[(str(row["stage"]), int(row["repeat"]),
                       str(row["task_id"]))] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return cache

    def _record(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[(row["stage"], row["repeat"], row["task_id"])] = row
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _map(self, jobs: Sequence[Any], fn: Callable[[Any, Any], None],
             *, label: str) -> None:
        """Run `fn(job, client)` over `jobs`, one client per worker thread."""
        if not jobs:
            return
        q: "queue.Queue" = queue.Queue()
        for j in jobs:
            q.put(j)
        n_workers = max(1, min(self.cfg.workers, len(jobs)))

        def _run(worker_id: int) -> None:
            client = CountingClient(self.client_factory(worker_id),
                                    self.stats, label)
            while True:
                try:
                    job = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    fn(job, client)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(f"[select w{worker_id}] "
                                     f"{type(exc).__name__}: {exc}\n")
                finally:
                    q.task_done()

        threads = [threading.Thread(target=_run, args=(i,), name=f"sel-w{i}")
                   for i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def run_stage(self, task_ids: Sequence[str], *, stage: str,
                  library: SkillLibrary,
                  forced: Sequence[str] | None = None,
                  repeats: int | None = None) -> Dict[str, Any]:
        """Run these items `repeats` times; per-item mean score and routing.

        Returns `{"per_item": {id: mean}, "repeats": [{id: row}, …]}`; a row
        keeps the chosen skills, so E3.7 can ask which items a skill attracted.

        The cache key is the caller's stage name PLUS a fingerprint of the
        library's documents: `forced/<sid>` alone is keyed by the id, so a
        second selection into the same `--out` after a stream was redone would
        decide the new skill from the scores of the old one. Two conditions
        that really are the same — an empty `library_selected/` and the NONE
        reference — still share their rows, because their libraries are equal.
        """
        repeats = self.cfg.repeats if repeats is None else repeats
        stage = f"{stage}@{condition_tag(library_docs(library))}"
        ids = [str(i) for i in task_ids]
        per_repeat: List[Dict[str, Dict[str, Any]]] = []
        for rep in range(1, repeats + 1):
            rows: Dict[str, Dict[str, Any]] = {}
            todo = []
            for tid in ids:
                hit = self._cache.get((stage, rep, tid))
                if hit is not None:
                    rows[tid] = hit
                else:
                    todo.append(tid)

            def _one(tid: str, client: Any, _rep: int = rep) -> None:
                task = self.by_id[tid]
                final_code = ""
                try:
                    kwargs: Dict[str, Any] = {}
                    if forced is not None:
                        kwargs["forced_skill_ids"] = list(forced)
                    rec = self.run_task_fn(task, client,
                                           executor_config=self.exec_cfg,
                                           library=library, **kwargs)
                    prediction = rec.prediction
                    status = rec.final_status
                    chosen = list(rec.chosen_skills or [])
                    attempts = list(getattr(rec, "code_attempts", None) or [])
                    final_code = str(attempts[-1]) if attempts else ""
                except Exception as exc:  # noqa: BLE001
                    prediction, status, chosen = "", "pipeline_crash", []
                    sys.stderr.write(f"[select] {tid}: "
                                     f"{type(exc).__name__}: {exc}\n")
                score_error = ""
                try:
                    score, _br = score_one(task["eval_metric"], prediction,
                                           task.get("ground_truth",
                                                    task.get("answer")))
                except Exception as exc:  # noqa: BLE001
                    score = 0.0
                    score_error = f"{type(exc).__name__}: {exc}"
                    sys.stderr.write(f"[select] {tid}: score_one failed: "
                                     f"{score_error}\n")
                row = {"stage": stage, "repeat": _rep, "task_id": tid,
                       "score": float(score), "status": str(status),
                       "score_error": score_error,
                       "chosen_skills": chosen,
                       "forced": (list(forced) if forced is not None else None),
                       # F5: the same three flags the eval records carry, so a
                       # selection run answers "did the skill make the code
                       # search the index?" without a second run.
                       **index_usage(final_code)}
                self._record(row)
                with self._lock:
                    rows[tid] = row

            t0 = time.time()
            self._map(todo, _one, label="pipeline")
            if todo:
                self.log(f"    [{stage}] repeat {rep}: {len(todo)} runs "
                         f"({time.time() - t0:.0f}s, "
                         f"{len(ids) - len(todo)} cached)")
            per_repeat.append(rows)

        per_item = {tid: mean([r[tid]["score"] for r in per_repeat
                               if tid in r])
                    for tid in ids
                    if any(tid in r for r in per_repeat)}
        health = self._stage_health(stage, per_repeat, expected=len(ids) * repeats)
        return {"stage": stage, "per_item": per_item, "repeats": per_repeat,
                "n": len(per_item), "health": health}

    # A run that ended in one of these produced no measurement at all: the
    # pipeline raised, or the answer could not be scored. Both are stored as
    # score 0.0, which is indistinguishable from a wrong answer — so they are
    # counted, and a stage that is full of them is refused instead of decided.
    INTERNAL_FAILURES = ("pipeline_error", "pipeline_crash")

    def _stage_health(self, stage: str,
                      per_repeat: Sequence[Dict[str, Dict[str, Any]]],
                      *, expected: int) -> Dict[str, Any]:
        """Status histogram of a stage; refuse the stage if it mostly failed.

        `selection.json` used to record nothing about how the runs ended, so a
        typo in `--databases` or an API outage produced every mean at 0.0, no
        cluster above the margin, and a well-formed REPORT.md that looked like
        a genuine negative result.
        """
        rows = [row for rep in per_repeat for row in rep.values()]
        hist: Dict[str, int] = {}
        for row in rows:
            key = str(row.get("status") or "(unknown)")
            hist[key] = hist.get(key, 0) + 1
        n_internal = sum(hist.get(s, 0) for s in self.INTERNAL_FAILURES)
        n_score_error = sum(1 for row in rows if row.get("score_error"))
        n_missing = max(0, expected - len(rows))
        bad = n_internal + n_score_error + n_missing
        health = {"stage": stage, "n_runs": len(rows), "expected": expected,
                  "status_counts": dict(sorted(hist.items())),
                  "n_internal_failures": n_internal,
                  "n_score_errors": n_score_error, "n_missing": n_missing,
                  "error_rate": round(bad / expected, 4) if expected else 0.0}
        with self._lock:
            self.stage_status[stage] = health
        if expected and bad / expected > self.cfg.max_error_rate:
            raise StageError(
                f"stage '{stage}': {bad} of {expected} runs did not produce a "
                f"measurement ({n_internal} pipeline failures, "
                f"{n_score_error} scoring failures, {n_missing} missing) — "
                f"above --max-error-rate {self.cfg.max_error_rate}. These are "
                f"stored as score 0.0 and would silently drop every skill; "
                f"check --databases, the API keys and the network, then re-run "
                f"(measured runs are cached in runs.jsonl).")
        return health

    # ----------------------------------------------------------------- E3.1 pool
    def build_pool(self) -> List[Dict[str, Any]]:
        """Copy every skill of every library into `pool/`, renaming clashes.

        `pool/` is rebuilt from scratch: it is re-read as a DIRECTORY for the
        routing manifest, for every forced run and by `write_selected`, so a
        leftover document from an earlier selection into the same `--out` would
        be routed to and then match no cluster member.
        """
        if self.pool_dir.exists():
            shutil.rmtree(self.pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._pool_docs: Dict[str, str] | None = None
        entries: List[Dict[str, Any]] = []
        self.libraries_summary = []
        self.dropped = []
        taken: Dict[str, str] = {}      # pool id -> original id
        for index, lib_dir in enumerate(self.libraries, start=1):
            lib = SkillLibrary(lib_dir)
            own_ids = lib.ids()
            n_before, n_dropped = len(entries), len(self.dropped)
            if not own_ids:
                why = ("the directory does not exist" if not Path(lib_dir).is_dir()
                       else "it holds no parsable skill document")
                raise SystemExit(
                    f"--library {lib_dir} contributes no skill: {why}. "
                    f"E3.1 pools three stream libraries; a typo would run a "
                    f"two-stream selection that looks like a three-stream one.")
            for skill in lib.skills():
                pool_id = skill.id
                if pool_id in taken:
                    pool_id = f"{skill.id}-{library_tag(index)}"
                    n = 2
                    while pool_id in taken:
                        pool_id = f"{skill.id}-{library_tag(index)}-{n}"
                        n += 1
                raw = (Path(lib_dir) / f"{skill.id}.md")
                text = (raw.read_text(encoding="utf-8") if raw.is_file()
                        else f"---\nid: {skill.id}\n"
                             f"description: {skill.description}\n---\n\n"
                             f"{skill.body}\n")
                doc = text if pool_id == skill.id else rewrite_skill_id(text,
                                                                        pool_id)
                # The check the skill faced when it was committed: the OTHER
                # skills of its OWN library. Linting against the whole pool
                # would let one stream's id turn another stream's prose into a
                # violation, which is an artefact of pooling, not a defect.
                res = mechanical_lint(doc, self.channel_literals, (),
                                      [s for s in own_ids if s != skill.id],
                                      doc_max_chars=self.cfg.doc_max_chars,
                                      snippet_max_lines=self.cfg.snippet_max_lines)
                entry = {"pool_id": pool_id, "original_id": skill.id,
                         "library_index": index, "library": str(lib_dir),
                         "description": skill.description,
                         "renamed": pool_id != skill.id,
                         "lint": res["verdict"],
                         "lint_rules": sorted({v["rule"]
                                               for v in res["violations"]}),
                         "violations": res["violations"]}
                if res["verdict"] == "fail":
                    # The id stays free: reserving it before the verdict is
                    # known made a LATER library's clean `alpha` ship as
                    # `alpha-s2`, a suffix implying a sibling that was dropped.
                    self.dropped.append(entry)
                    self.log(f"  pool: DROP {pool_id} "
                             f"({', '.join(entry['lint_rules'])})")
                    continue
                taken[pool_id] = skill.id
                (self.pool_dir / f"{pool_id}.md").write_text(doc,
                                                             encoding="utf-8")
                entries.append(entry)
            self.libraries_summary.append(
                {"index": index, "library": str(lib_dir),
                 "n_skills": len(own_ids), "n_pooled": len(entries) - n_before,
                 "n_dropped": len(self.dropped) - n_dropped})
            self.log(f"  library s{index}: {len(own_ids)} skills, "
                     f"{len(entries) - n_before} pooled, "
                     f"{len(self.dropped) - n_dropped} dropped — {lib_dir}")
        self.pool = entries
        self._pool_docs = None
        self.log(f"  pool: {len(entries)} skills "
                 f"({len(self.dropped)} dropped by the mechanical lint)")
        return entries

    def pool_library(self) -> SkillLibrary:
        return SkillLibrary(self.pool_dir)

    def pool_docs(self) -> Dict[str, str]:
        """The pooled documents, by pool id (the cache key of every stage)."""
        cached = getattr(self, "_pool_docs", None)
        if cached is None:
            cached = {p.stem: p.read_text(encoding="utf-8")
                      for p in sorted(self.pool_dir.glob("*.md"))}
            self._pool_docs = cached
        return cached

    def pool_tag(self) -> str:
        """Fingerprint of "the pool exactly as it is now", documents included.

        Every stage that a pooled document can influence carries it: a forced
        run generates from that document and plans against this manifest, and
        the routing that defines `V_c` is this manifest's answer.
        """
        return condition_tag(self.pool_docs())

    # -------------------------------------------------------------- E3.2 cluster
    def render_pool_for_cluster(self) -> str:
        """id, description and "When to use" of every pooled skill."""
        lib = self.pool_library()
        blocks: List[str] = []
        for entry in self.pool:
            skill = lib.get(entry["pool_id"])
            if skill is None:
                continue
            when = _section(skill.body, "When to use")
            blocks.append(f"### {skill.id}\ndescription: {skill.description}\n"
                          f"when to use: {when or '(not stated)'}")
        return "\n\n".join(blocks) if blocks else "(none)"

    def cluster_pool(self) -> List[Tuple[str, List[str]]]:
        """One cluster call; every pooled id lands in exactly one cluster."""
        ids = [e["pool_id"] for e in self.pool]
        if not ids:
            return []
        agents = Agents(self.client_factory(0), prompts_dir=self.prompts_dir,
                        stats=self.stats,
                        max_tokens=self.cfg.distill_max_tokens)
        parsed, _prompt, raws, reasons = agents.cluster_full(
            skills=self.render_pool_for_cluster())
        (self.out / "cluster_raw.json").write_text(
            json.dumps({"parsed": parsed, "raw": raws,
                        "finish_reasons": reasons}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        if any(r == "length" for r in reasons):
            # `assign_clusters` turns an unparsable reply into all-singletons,
            # which silently disables the cross-stream comparison E3.5 exists
            # for. A reply cut off at the ceiling is an error, not a verdict.
            raise StageError(
                f"the cluster reply was cut off at max_tokens "
                f"({self.cfg.distill_max_tokens}) for {len(ids)} pooled "
                f"skills; raising --distill-max-tokens or pooling fewer "
                f"skills is required — treating it as 'all singletons' would "
                f"silently disable the duplicate comparison.")
        clusters = assign_clusters(parsed, ids)
        self.log(f"  clusters: {len(clusters)} "
                 f"({', '.join(str(len(m)) for _i, m in clusters)})")
        return clusters

    # ---------------------------------------------------------------- E3.3 route
    def route_valid(self, task_ids: Sequence[str]) -> Dict[str, List[str]]:
        """Planner-only routing of every VALID item with the pooled manifest.

        Cached routes are only reused when the pooled manifest is byte-for-byte
        the one they were measured with, a failed planner call is retried once
        and then fatal, and an `ALL` pick is expanded to the pooled ids — an
        item that is silently missing from `routes` shrinks every `V_c` it
        belonged to and can push a cluster below the `|V_c| >= 5` floor.
        """
        tag = self.pool_tag()
        pool_ids = [e["pool_id"] for e in self.pool] or list(
            self.pool_library().ids())
        cached: Dict[str, List[str]] = {}
        if self.routes_path.is_file():
            try:
                blob = json.loads(self.routes_path.read_text(encoding="utf-8"))
                if (isinstance(blob, dict)
                        and blob.get("manifest_tag") == tag
                        and isinstance(blob.get("routes"), dict)):
                    cached = {str(k): list(v)
                              for k, v in blob["routes"].items()}
                else:
                    self.log("    routes.json was measured with a different "
                             "pool — re-routing")
            except (json.JSONDecodeError, AttributeError, TypeError):
                cached = {}
        manifest = self.pool_library().manifest()
        picks: Dict[str, List[str]] = {tid: cached[tid] for tid in task_ids
                                       if tid in cached}
        failures: Dict[str, str] = {}

        def _one(tid: str, client: Any) -> None:
            try:
                ids = self.route_fn(self.by_id[tid], manifest, client)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    failures[tid] = f"{type(exc).__name__}: {exc}"
                return
            with self._lock:
                failures.pop(tid, None)
                picks[tid] = expand_all(ids, pool_ids)

        for attempt in (1, 2):
            todo = [tid for tid in task_ids if tid not in picks]
            if not todo:
                break
            if attempt == 2:
                self.log(f"    routing: retrying {len(todo)} failed call(s)")
            self._map(todo, _one, label="planner_route")

        self.route_failures = sorted(failures)
        self.routes_path.write_text(
            json.dumps({"manifest_tag": tag, "routes": picks}, indent=2),
            encoding="utf-8")
        if failures:
            raise StageError(
                f"{len(failures)} of {len(task_ids)} planner routing calls "
                f"failed twice (e.g. {sorted(failures)[0]}: "
                f"{failures[sorted(failures)[0]]}). Those items would silently "
                f"leave every V_c; fix the outage and re-run (the routes "
                f"already measured are cached in routes.json).")
        return picks

    @staticmethod
    def cluster_items(members: Sequence[str], routes: Dict[str, List[str]],
                      task_ids: Sequence[str]) -> List[str]:
        """`V_c`: the VALID items the planner routes to ANY cluster member."""
        member_set = set(members)
        return [tid for tid in task_ids
                if member_set & set(routes.get(tid, []))]

    # ------------------------------------------------------ H9 cost cap
    def subsample_items(self, items: Sequence[str], *, cluster: int
                        ) -> Tuple[List[str], Dict[str, Any] | None]:
        """`V_c` as it will be MEASURED, plus the record of what was cut.

        Validating a cluster costs `|M_c| · |V_c| · repeats` pipeline runs, and
        round 2's largest cluster alone (4 members × 150 items × 2) was 1 200
        of the 3 000 runs behind a 45 RMB command. Above the cap the cluster is
        judged on a seeded random subset instead — the same subset for the NONE
        reference and for every member, because the decision is a DIFFERENCE
        and comparing two members on two different item sets would be noise
        with a margin bolted on.

        The seed is `--seed` mixed with the cluster number, so the choice is
        reproducible from `selection.json` alone, two clusters do not get
        correlated subsets, and a re-run into the same `--out` reuses the
        cached rows instead of paying for a fresh draw. Ordering follows
        `task_ids`, not the draw, so the subset reads like the item list it
        came from.
        """
        cap = int(self.cfg.max_items_per_cluster)
        full = [str(t) for t in items]
        if cap <= 0 or len(full) <= cap:
            return full, None
        rng = random.Random(f"{self.cfg.seed}|cluster{cluster}")
        picked = set(rng.sample(full, cap))
        subset = [tid for tid in full if tid in picked]
        info = {"cluster": cluster, "n_routed": len(full), "n_evaluated": cap,
                "seed": int(self.cfg.seed),
                "rng_seed": f"{self.cfg.seed}|cluster{cluster}",
                "max_items_per_cluster": cap,
                "dropped": [tid for tid in full if tid not in picked]}
        return subset, info

    # --------------------------------------------------------------- E3.5 keep
    def validate_clusters(self, clusters: Sequence[Tuple[str, List[str]]],
                          routes: Dict[str, List[str]],
                          task_ids: Sequence[str],
                          none_per_item: Dict[str, float]
                          ) -> List[Dict[str, Any]]:
        """Best member per cluster, kept only if it beats NONE by the margin."""
        pool_lib = self.pool_library()
        decisions: List[Dict[str, Any]] = []
        for n, (intent, members) in enumerate(clusters, start=1):
            routed = self.cluster_items(members, routes, task_ids)
            # H9: |V_c| is what the planner routed; `items` is what is paid for.
            items, subsample = self.subsample_items(routed, cluster=n)
            record: Dict[str, Any] = {
                "cluster": n, "shared_intent": intent, "members": list(members),
                "n_items": len(items), "items": items, "kept": None,
                "n_routed_items": len(routed), "routed_items": routed,
                "subsample": subsample,
                "members_mean": {}, "none_mean": None, "margin": self.cfg.select_margin,
            }
            if len(routed) < self.cfg.min_valid_items:
                record["decision"] = "not_validated"
                record["reason"] = (f"|V_c| = {len(routed)} < "
                                    f"{self.cfg.min_valid_items}")
                self.log(f"  cluster {n} ({len(members)} members): "
                         f"|V_c| = {len(routed)} → nothing kept")
                decisions.append(record)
                continue
            if subsample is not None:
                self.log(f"  cluster {n}: |V_c| = {len(routed)} > "
                         f"--max-items-per-cluster "
                         f"{self.cfg.max_items_per_cluster} → validated on a "
                         f"seeded random subset of {len(items)} "
                         f"(seed {subsample['rng_seed']})")
            none_mean = mean([none_per_item[t] for t in items
                              if t in none_per_item])
            record["none_mean"] = round(none_mean, 4)
            for sid in members:
                res = self.run_stage(items, stage=f"forced/{sid}",
                                     library=pool_lib, forced=[sid])
                record["members_mean"][sid] = round(
                    mean([res["per_item"][t] for t in items
                          if t in res["per_item"]]), 4)
            best = max(members, key=lambda s: (record["members_mean"].get(s, 0.0),
                                               -members.index(s)))
            best_mean = record["members_mean"].get(best, 0.0)
            record["best"] = best
            record["best_mean"] = best_mean
            record["delta_vs_none"] = round(best_mean - none_mean, 4)
            if best_mean - none_mean >= self.cfg.select_margin:
                record["decision"] = "keep"
                record["kept"] = best
            else:
                record["decision"] = "drop"
                record["reason"] = (f"best − NONE = {best_mean - none_mean:+.4f} "
                                    f"< {self.cfg.select_margin}")
            self.log(f"  cluster {n} ({len(members)} members, "
                     f"|V_c| = {len(routed)}"
                     + (f", {len(items)} evaluated" if subsample else "")
                     + f"): best {best} "
                     f"{best_mean:.4f} vs NONE {none_mean:.4f} "
                     f"→ {record['decision']}")
            decisions.append(record)
        return decisions

    def write_selected(self, kept: Sequence[str]) -> SkillLibrary:
        """`library_selected/` holds exactly the kept skills (E3.6)."""
        if self.selected_dir.exists():
            shutil.rmtree(self.selected_dir)
        self.selected_dir.mkdir(parents=True, exist_ok=True)
        for sid in kept:
            src = self.pool_dir / f"{sid}.md"
            if src.is_file():
                shutil.copy2(src, self.selected_dir / src.name)
        return SkillLibrary(self.selected_dir)

    # ------------------------------------------------------- E3.7 do no harm
    @staticmethod
    def attributed_scores(stage: Dict[str, Any], skill_id: str,
                          library_ids: Sequence[str] | None = None
                          ) -> Dict[str, float]:
        """Per item: the mean score of the repeats that actually loaded `s`.

        A repeat in which the planner did not pick `s` says nothing about `s`,
        so it is left out instead of being averaged in. A plan that says `ALL`
        loads every skill of `library_ids`, so `s` is charged with it — before,
        such an item escaped the one gate that catches a harmful skill.
        """
        per_item: Dict[str, List[float]] = {}
        for rows in stage["repeats"]:
            for tid, row in rows.items():
                chosen = expand_all(row.get("chosen_skills") or [], library_ids)
                if skill_id in chosen:
                    per_item.setdefault(tid, []).append(float(row["score"]))
        return {tid: mean(xs) for tid, xs in per_item.items() if xs}

    def do_no_harm(self, kept: List[str], task_ids: Sequence[str],
                   none_per_item: Dict[str, float]) -> Dict[str, Any]:
        """Run VALID with the selected library; drop a skill that harms (E3.7).

        One repeat of the whole check, exactly as E3 allows: round 2 measures
        the library after the drops and reports — it does not drop again.
        """
        rounds: List[Dict[str, Any]] = []
        current = list(kept)
        for rnd in (1, 2):
            library = self.write_selected(current)
            # An empty selected library IS the NONE condition, so it shares the
            # reference's rows instead of paying for them a second time.
            docs = self.pool_docs()
            stage_name = ("none" if not current else "selected/" + condition_tag(
                {sid: docs.get(sid, "") for sid in current}))
            stage = self.run_stage(task_ids, stage=stage_name, library=library)
            overall = mean([stage["per_item"][t] for t in task_ids
                            if t in stage["per_item"]])
            per_skill: List[Dict[str, Any]] = []
            offenders: List[str] = []
            for sid in current:
                got = self.attributed_scores(stage, sid, current)
                items = sorted(got)
                with_mean = mean([got[t] for t in items])
                none_mean = mean([none_per_item[t] for t in items
                                  if t in none_per_item])
                delta = with_mean - none_mean
                harms = (len(items) >= self.cfg.harm_min_items
                         and delta < -self.cfg.harm_margin)
                per_skill.append({
                    "skill_id": sid, "n_routed": len(items),
                    "mean_with": round(with_mean, 4),
                    "mean_none": round(none_mean, 4),
                    "delta": round(delta, 4), "harms": harms})
                if harms:
                    offenders.append(sid)
            rounds.append({"round": rnd, "library": list(current),
                           "overall_mean": round(overall, 4),
                           "per_skill": per_skill, "offenders": offenders})
            self.log(f"  do-no-harm round {rnd}: overall {overall:.4f} "
                     f"(NONE {mean([none_per_item[t] for t in task_ids if t in none_per_item]):.4f})"
                     + (f", dropping {', '.join(offenders)}" if offenders and rnd == 1
                        else ""))
            if rnd == 2 or not offenders:
                break
            current = [s for s in current if s not in offenders]
        final_library = self.write_selected(current)
        return {"rounds": rounds, "final_library": list(current),
                "n_skills": len(final_library.ids())}

    # ------------------------------------------------------------------- driver
    def run(self, task_ids: Sequence[str]) -> Dict[str, Any]:
        t0 = time.time()
        ids = [str(t) for t in task_ids if str(t) in self.by_id]
        missing = [str(t) for t in task_ids if str(t) not in self.by_id]
        self.log(f"VALID items: {len(ids)}"
                 + (f" ({len(missing)} ids not in tasks.json)" if missing else ""))

        self.log("[1/7] pool")
        self.build_pool()
        self.log("[2/7] cluster")
        clusters = self.cluster_pool()
        self.log("[3/7] route")
        routes = self.route_valid(ids)
        self.log("[4/7] NONE reference")
        none_stage = self.run_stage(ids, stage="none",
                                    library=SkillLibrary(self.empty_dir))
        none_per_item = none_stage["per_item"]
        none_mean = mean([none_per_item[t] for t in ids if t in none_per_item])
        self.log(f"    NONE overall {none_mean:.4f}")
        self.log("[5/7] validate clusters")
        decisions = self.validate_clusters(clusters, routes, ids, none_per_item)
        kept = [d["kept"] for d in decisions if d.get("kept")]
        self.log(f"[6/7] selected library: {len(kept)} skills "
                 + (", ".join(kept) if kept else "(empty)"))
        self.log("[7/7] do no harm")
        harm = self.do_no_harm(kept, ids, none_per_item)

        n_routed_nowhere = sum(1 for t in ids if not routes.get(t))
        selection = {
            "config": self.cfg.to_dict(),
            "libraries": [str(p) for p in self.libraries],
            "libraries_summary": self.libraries_summary,
            "pool_tag": self.pool_tag(),
            "stage_health": [self.stage_status[k]
                             for k in sorted(self.stage_status)],
            "n_route_failures": len(self.route_failures),
            "route_failures": self.route_failures,
            "n_valid": len(ids), "valid_ids_not_in_tasks": missing,
            "pool": self.pool, "pool_dropped": self.dropped,
            "clusters": [{"cluster": n, "shared_intent": i, "members": m}
                         for n, (i, m) in enumerate(clusters, start=1)],
            "routes": routes, "n_routed_nowhere": n_routed_nowhere,
            "none_mean": round(none_mean, 4),
            "none_per_item": {k: round(v, 4) for k, v in none_per_item.items()},
            "decisions": decisions,
            # H9: the record of the cost cap. Empty when no cluster was over
            # it, so a selection that paid for every routed item says so.
            "max_items_per_cluster": self.cfg.max_items_per_cluster,
            "subsampling": [d["subsample"] for d in decisions
                            if d.get("subsample")],
            "kept_after_validation": kept,
            "do_no_harm": harm,
            "final_library": harm["final_library"],
            "valid_mean_none": round(none_mean, 4),
            "valid_mean_selected": (harm["rounds"][-1]["overall_mean"]
                                    if harm["rounds"] else None),
            "llm_calls": self.stats.snapshot(),
            "exec_concurrency": EXEC_CONCURRENCY,
            "seconds": round(time.time() - t0, 1),
        }
        (self.out / "selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (self.out / "REPORT.md").write_text(render_report(selection),
                                            encoding="utf-8")
        return selection


def _default_route(task: Dict[str, Any], manifest: str,
                   client: Any) -> List[str]:
    """Planner-only call (no code generation) — same as the routing check."""
    schema = build_schema_text(task["id"])
    return plan_mod.plan(task["question"], schema, manifest, client).skill_ids


# --------------------------------------------------------------------------- report
def render_report(sel: Dict[str, Any]) -> str:
    cfg = sel["config"]
    lines = ["# Validation-based selection (ADDENDUM E3)", "",
             f"VALID items: {sel['n_valid']} — repeats {cfg['repeats']}, "
             f"select margin {cfg['select_margin']}, "
             f"|V_c| ≥ {cfg['min_valid_items']}, "
             f"harm margin {cfg['harm_margin']} on ≥ {cfg['harm_min_items']} items, "
             f"at most {cfg.get('max_items_per_cluster')} items evaluated per "
             f"cluster (seed {cfg.get('seed')}).",
             "", "## Pool", ""]
    for lib in sel.get("libraries_summary") or []:
        lines.append(f"- s{lib['index']}: {lib['n_skills']} skills, "
                     f"{lib['n_pooled']} pooled, {lib['n_dropped']} dropped "
                     f"— `{lib['library']}`")
    lines += ["",
              "| pool id | from | original id | lint |", "|---|---:|---|---|"]
    for e in sel["pool"]:
        lines.append(f"| {e['pool_id']} | s{e['library_index']} | "
                     f"{e['original_id']} | {e['lint']} |")
    if sel["pool_dropped"]:
        lines += ["", "Dropped by the mechanical lint:", ""]
        for e in sel["pool_dropped"]:
            lines.append(f"- `{e['pool_id']}` (s{e['library_index']}): "
                         f"{', '.join(e['lint_rules'])}")
    lines += ["", "## Clusters and decisions", "",
              f"NONE reference over all VALID items: **{sel['none_mean']:.4f}**; "
              f"{sel['n_routed_nowhere']} items route nowhere.", "",
              "| # | members | \\|V_c\\| | member means | best | NONE on V_c | "
              "Δ | decision |", "|---:|---|---:|---|---|---:|---:|---|"]
    for d in sel["decisions"]:
        means = ", ".join(f"{k} {v:.4f}" for k, v in
                          sorted(d["members_mean"].items(),
                                 key=lambda kv: -kv[1])) or "—"
        # H9: routed first — the cell must never claim a cluster was smaller
        # than it is just because the cap cut what was measured.
        n_routed = d.get("n_routed_items", d["n_items"])
        v_cell = (f"{n_routed} ({d['n_items']} eval.)"
                  if d.get("subsample") else str(n_routed))
        lines.append(
            f"| {d['cluster']} | {', '.join(d['members'])} | {v_cell} | "
            f"{means} | {d.get('best', '—')} | "
            f"{d['none_mean'] if d['none_mean'] is not None else '—'} | "
            f"{d.get('delta_vs_none', '—')} | {d['decision']}"
            + (f" ({d['reason']})" if d.get("reason") else "") + " |")
    subs = sel.get("subsampling") or []
    if subs:
        lines += ["", "### Subsampled clusters (`--max-items-per-cluster` "
                      f"{cfg.get('max_items_per_cluster')})", "",
                  "The cluster's member means and its NONE column above are "
                  "measured on this seeded random subset of the routed items, "
                  "the same subset for NONE and for every member. The "
                  "do-no-harm stage below is NOT subsampled.", "",
                  "| cluster | routed | evaluated | rng seed |",
                  "|---:|---:|---:|---|"]
        for s in subs:
            lines.append(f"| {s['cluster']} | {s['n_routed']} | "
                         f"{s['n_evaluated']} | `{s['rng_seed']}` |")
    lines += ["", "## Do no harm under real routing", ""]
    for rnd in sel["do_no_harm"]["rounds"]:
        lines += [f"### Round {rnd['round']} — overall VALID mean "
                  f"{rnd['overall_mean']:.4f} "
                  f"(NONE {sel['none_mean']:.4f})", "",
                  "| skill | items routed to it | with skill | NONE | Δ | harms |",
                  "|---|---:|---:|---:|---:|---|"]
        for s in rnd["per_skill"]:
            lines.append(f"| {s['skill_id']} | {s['n_routed']} | "
                         f"{s['mean_with']:.4f} | {s['mean_none']:.4f} | "
                         f"{s['delta']:+.4f} | {'YES' if s['harms'] else 'no'} |")
        if rnd["offenders"]:
            lines += ["", f"Offenders: {', '.join(rnd['offenders'])}"
                          + (" — dropped." if rnd["round"] == 1
                             else " — reported only (one repeat allowed).")]
        lines.append("")
    health = sel.get("stage_health") or []
    if health:
        lines += ["## How the runs ended", "",
                  "A pipeline failure and an unscorable answer are both stored "
                  "as score 0.0; a stage full of them would look like a "
                  "genuine negative result.", "",
                  "| stage | runs | statuses | internal failures | "
                  "scoring failures | missing | error rate |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for h in health:
            counts = ", ".join(f"{k} {v}" for k, v in
                               (h.get("status_counts") or {}).items()) or "—"
            lines.append(
                f"| {h['stage']} | {h['n_runs']} | {counts} | "
                f"{h['n_internal_failures']} | {h['n_score_errors']} | "
                f"{h['n_missing']} | {h['error_rate']:.4f} |")
        lines += ["", f"Planner routing calls that failed twice: "
                      f"{sel.get('n_route_failures', 0)}.", ""]
    lines += ["## Result", "",
              f"- selected library: {len(sel['final_library'])} skill(s): "
              + (", ".join(f"`{s}`" for s in sel["final_library"]) or "(empty)"),
              f"- VALID mean NONE {sel['valid_mean_none']:.4f} → selected "
              f"{sel['valid_mean_selected']}",
              f"- LLM calls: {sel['llm_calls'].get('calls')} "
              f"{sel['llm_calls'].get('by_agent')}",
              f"- wall time: {sel['seconds']} s, "
              f"exec concurrency {sel['exec_concurrency']}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def main(argv: List[str] | None = None, *,
         run_task_fn: Callable[..., Any] = _default_run_task,
         route_fn: Callable[..., List[str]] | None = None,
         client_factory: Callable[[int], Any] | None = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(prog="python -m cold_start_stream.select",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", action="append", required=True, dest="libraries",
                   help="a stream's library_final directory (repeatable).")
    p.add_argument("--tasks", required=True, help="tasks.json of the train set.")
    p.add_argument("--databases", required=True,
                   help="COPY of the VALID databases (never the read-only tree).")
    p.add_argument("--ids-file", required=True,
                   help="JSON list of the VALID task ids.")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(_CONFIG_PATH))
    defaults = SelectConfig()
    for f in fields(SelectConfig):
        p.add_argument("--" + f.name.replace("_", "-"),
                       type=type(getattr(defaults, f.name)), default=None,
                       dest=f.name, help=f"(default: {getattr(defaults, f.name)})")
    args = p.parse_args(argv)
    # ADDENDUM H5: a non-SAX symbolization against the default column
    # name loses every index read, silently. Abort before the run.
    check_symbolization_env()

    cfg = SelectConfig()
    for f in fields(SelectConfig):
        value = getattr(args, f.name)
        if value is not None:
            setattr(cfg, f.name, value)

    tasks_json = Path(args.tasks).resolve()
    paths_mod.TASKS_JSON = tasks_json
    tasks = json.loads(tasks_json.read_text(encoding="utf-8"))
    valid_ids = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
    if not isinstance(valid_ids, list) or not valid_ids:
        raise SystemExit(f"--ids-file {args.ids_file} is not a non-empty list")
    known = {str(t["id"]) for t in tasks}
    paths_mod.DATABASES_DIR = check_databases_dir(
        args.databases,
        required_ids=[str(i) for i in valid_ids if str(i) in known])

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    llm_cfg = config["llm"]
    exec_cfg = {
        "max_regenerations": config["executor"]["max_regenerations"],
        "timeout_seconds": config["executor"]["timeout_seconds"],
        "shape_rules": config["shape_rules"],
    }
    if client_factory is None:
        api_keys = load_api_keys(llm_cfg["api_keys_file"])

        def client_factory(worker_id: int) -> Any:  # noqa: F811
            return DeepSeekClient(
                api_key=api_keys[worker_id % len(api_keys)],
                base_url=llm_cfg["base_url"], model=llm_cfg["model"],
                temperature=llm_cfg["temperature"],
                max_tokens=llm_cfg["max_tokens"],
                thinking_mode=llm_cfg.get("thinking_mode", "disabled"),
                max_attempts=llm_cfg["http_max_attempts"],
                backoff_seconds=llm_cfg["http_backoff_seconds"],
            )

    selector = Selector(libraries=args.libraries, tasks=tasks, out=args.out,
                        exec_cfg=exec_cfg, client_factory=client_factory,
                        cfg=cfg, run_task_fn=run_task_fn, route_fn=route_fn)
    print(f"libraries : {', '.join(args.libraries)}")
    print(f"valid     : {len(valid_ids)} ids")
    print(f"workers   : {cfg.workers}, exec concurrency {EXEC_CONCURRENCY}")
    selection = selector.run(valid_ids)
    print("\n" + (Path(args.out) / "REPORT.md").read_text(encoding="utf-8"))
    return selection


if __name__ == "__main__":
    main()
