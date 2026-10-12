"""CLI: score a skill library on a held-out subset of the benchmark.

    python -m cold_start_stream.eval --tasks <tasks.json> --databases <dir>
        --library <dir | NONE | PACKAGED> --subset <json of ids>
        --out <dir> [--workers W]
        [--category <name|code>[,...]] [--ids-file <json of ids>]
        [--force-skill <skill id>[,...]]

Evaluation may use labels (the cold start may not): the table is grouped by
`category`.

`--force-skill` is a diagnostic: the planner still writes the plan (and its own
pick is kept in `planner_skills`), but the generator is handed exactly the
named skills. `--category` restricts the subset to benchmark categories or
sub-tasks, by full name or by the short code of `CATEGORY_ORDER`.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import yaml

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from sonar_ts import _paths as paths_mod  # noqa: E402
from sonar_ts.evaluator import CATEGORY_ORDER, score_one  # noqa: E402
from sonar_ts.executor import EXEC_CONCURRENCY  # noqa: E402
from sonar_ts.llm import DeepSeekClient, load_api_keys  # noqa: E402
from sonar_ts.pipeline import run_task  # noqa: E402
from sonar_ts.skills import SkillLibrary  # noqa: E402

from .guards import (check_databases_dir,  # noqa: E402
                     check_symbolization_env)
from .index_usage import FLAGS as INDEX_FLAGS, index_usage  # noqa: E402

_CONFIG_PATH = _FRAMEWORK_ROOT / "configs" / "online.yaml"
_PACKAGED_LIBRARY = _FRAMEWORK_ROOT / "sonar_ts" / "skills" / "library"


def resolve_library(spec: str, out: Path) -> SkillLibrary:
    if spec.upper() == "NONE":
        empty = out / "_empty_library"
        empty.mkdir(parents=True, exist_ok=True)
        return SkillLibrary(empty)
    if spec.upper() == "PACKAGED":
        return SkillLibrary(_PACKAGED_LIBRARY)
    return SkillLibrary(Path(spec).resolve())


def parse_csv_option(value: str | None) -> List[str] | None:
    """`"a, b ,,c"` -> `["a", "b", "c"]`; `None`/blank -> `None`."""
    if value is None:
        return None
    items = [piece.strip() for piece in value.split(",")]
    items = [piece for piece in items if piece]
    return items or None


def forced_key(forced: Sequence[str] | None) -> tuple[str, ...] | None:
    """The identity of a run condition: the forced skill list, or `None`."""
    return None if forced is None else tuple(forced)


def select_categories(tasks: List[Dict[str, Any]],
                      names: Iterable[str]) -> List[Dict[str, Any]]:
    """Keep the tasks whose `category` or `subtask` is one of `names`.

    A name may be a full category name ("Composite Trend"), its short code
    ("CT") or a sub-task name ("Temporal Localization"); matching ignores case.
    An unknown name is fatal — it would silently evaluate nothing.
    """
    alias: Dict[str, set[str]] = {}

    def _add(key: str, cat_or_sub: str, field: str) -> None:
        alias.setdefault(key.lower(), set()).add(f"{field}:{cat_or_sub}")

    for cat, _level, code in CATEGORY_ORDER:
        _add(cat, cat, "category")
        _add(code, cat, "category")
    for t in tasks:
        _add(str(t.get("category", "")), str(t.get("category", "")), "category")
        _add(str(t.get("subtask", "")), str(t.get("subtask", "")), "subtask")
    alias.pop("", None)

    wanted: set[str] = set()
    unknown: List[str] = []
    for name in names:
        hit = alias.get(name.strip().lower())
        if hit is None:
            unknown.append(name)
        else:
            wanted |= hit
    if unknown:
        known = {cat for cat, _l, _c in CATEGORY_ORDER}
        known |= {code for _cat, _l, code in CATEGORY_ORDER}
        known |= {str(t.get("subtask")) for t in tasks if t.get("subtask")}
        raise SystemExit(
            "unknown --category value(s): " + ", ".join(sorted(unknown)) +
            "\nknown: " + ", ".join(sorted(known)))
    return [t for t in tasks
            if f"category:{t.get('category')}" in wanted
            or f"subtask:{t.get('subtask')}" in wanted]


def load_done(records: Path,
              forced: Sequence[str] | None = None) -> set[str]:
    """Task ids already recorded *under the same forced-skill condition*.

    Records written before `--force-skill` existed carry no `forced_skills`
    field; they count as the unforced condition, so a plain resume behaves
    exactly as before.
    """
    want = forced_key(forced)
    done: set[str] = set()
    if not records.is_file():
        return done
    for line in records.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            got = row.get("forced_skills")
            if forced_key(got if got is None else list(got)) != want:
                continue
            done.add(row["task_id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def status_histogram(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """How the runs ended, by executor status (D1).

    `timeout` is the number D7 reports per condition; it is derivable from
    `records.jsonl`, but a table that states the mean and nothing else invites
    a comparison between conditions that ran under different settings.
    """
    out: Dict[str, int] = {}
    for r in rows:
        key = str(r.get("status") or "(unknown)")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def index_usage_cell(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The three F5 rates over the rows that recorded a flag.

    Only rows that carry the flags count: a `records.jsonl` written before F5
    has none, and counting those as "did not use the index" would invent a
    number the run never measured.
    """
    known = [r for r in rows if any(f in r for f in INDEX_FLAGS)]
    n = len(known)
    cell: Dict[str, Any] = {"n": n}
    for flag in INDEX_FLAGS:
        hits = sum(1 for r in known if bool(r.get(flag)))
        cell[flag] = round(hits / n, 4) if n else 0.0
    return cell


def build_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, List[float]] = {}
    by_cat_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(float(r["score"]))
        by_cat_rows.setdefault(r["category"], []).append(r)
    cells = []
    index_cells = []
    for cat, level, code in CATEGORY_ORDER:
        xs = by_cat.get(cat, [])
        cells.append({"category": cat, "code": code, "level": level,
                      "n": len(xs),
                      "avg": round(sum(xs) / len(xs), 4) if xs else 0.0,
                      "status_counts": status_histogram(
                          by_cat_rows.get(cat, []))})
        index_cells.append({"category": cat, "code": code,
                            **index_usage_cell(by_cat_rows.get(cat, []))})
    all_scores = [float(r["score"]) for r in rows]
    return {"by_category": cells,
            "overall": {"n": len(all_scores),
                        "avg": round(sum(all_scores) / len(all_scores), 4)
                        if all_scores else 0.0},
            # F5: the diagnostic the whole round exists for.
            "index_usage_by_category": index_cells,
            "index_usage_overall": index_usage_cell(rows),
            "status_counts": status_histogram(rows)}


def render_markdown(table: Dict[str, Any], title: str) -> str:
    lines = [f"# {title}", "",
             "| Category | Code | Level | N | Score |",
             "|---|---|---|---:|---:|"]
    for c in table["by_category"]:
        lines.append(f"| {c['category']} | {c['code']} | {c['level']} | "
                     f"{c['n']} | {c['avg']:.4f} |")
    o = table["overall"]
    lines += ["", f"**Overall**: N = {o['n']}, score = {o['avg']:.4f}", ""]
    index_cells = table.get("index_usage_by_category")
    if index_cells:
        lines += ["## Index usage (share of tasks whose final code did it)", "",
                  "| Category | Code | N | feature table | sax | sax pattern |",
                  "|---|---|---:|---:|---:|---:|"]
        for c in index_cells:
            lines.append(f"| {c['category']} | {c['code']} | {c['n']} | "
                         f"{c['uses_feature_table']:.4f} | "
                         f"{c['uses_sax']:.4f} | "
                         f"{c['uses_sax_pattern']:.4f} |")
        t = table.get("index_usage_overall") or {}
        if t:
            lines += ["", f"**Overall index usage**: N = {t.get('n', 0)}, "
                          f"feature table {t.get('uses_feature_table', 0.0):.4f}, "
                          f"sax {t.get('uses_sax', 0.0):.4f}, "
                          f"sax pattern {t.get('uses_sax_pattern', 0.0):.4f}", ""]
    counts = table.get("status_counts") or {}
    if counts:
        lines += ["**Executor status**: "
                  + ", ".join(f"{k} {v}" for k, v in counts.items()),
                  ""]
    settings = [f"{k} = {table[k]}" for k in
                ("workers", "exec_concurrency", "exec_timeout_seconds")
                if table.get(k) is not None]
    if settings:
        lines += ["**Settings**: " + ", ".join(settings), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None, *,
         run_task_fn: Callable[..., Any] = run_task,
         client_factory: Callable[[int], Any] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m cold_start_stream.eval",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", required=True)
    p.add_argument("--databases", required=True)
    p.add_argument("--library", required=True,
                   help="library directory, or NONE, or PACKAGED")
    p.add_argument("--subset", default=None,
                   help="JSON list of task ids to evaluate (default: all).")
    p.add_argument("--ids-file", default=None,
                   help="further restrict to the task ids in this JSON list.")
    p.add_argument("--category", default=None,
                   help="comma-separated categories/sub-tasks to keep "
                        "(full name or short code, e.g. 'CT,Insight Synthesis').")
    p.add_argument("--force-skill", default=None,
                   help="comma-separated skill ids handed to the generator for "
                        "every task; the planner still writes the plan.")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--config", default=str(_CONFIG_PATH))
    args = p.parse_args(argv)
    # ADDENDUM H5: a non-SAX symbolization against the default column
    # name loses every index read, silently. Abort before the run.
    check_symbolization_env()

    forced = parse_csv_option(args.force_skill)
    categories = parse_csv_option(args.category)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks_json = Path(args.tasks).resolve()
    paths_mod.TASKS_JSON = tasks_json
    paths_mod.DATABASES_DIR = check_databases_dir(args.databases)

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
                extra_body=llm_cfg.get("extra_body"),
                api_style=llm_cfg.get("api_style"),
            )

    workers = args.workers or config.get("concurrency", {}).get("workers", 16)
    # More workers than keys is allowed: keys are handed out round-robin and
    # the API serves several concurrent requests per key.
    workers = max(1, workers)

    library = resolve_library(args.library, out)
    if forced:
        missing = [s for s in forced if s not in set(library.ids())]
        if missing:
            raise SystemExit(f"--force-skill: not in the library: "
                             f"{', '.join(missing)}")
    tasks = json.loads(tasks_json.read_text(encoding="utf-8"))
    if args.subset:
        wanted = set(json.loads(Path(args.subset).read_text(encoding="utf-8")))
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.ids_file:
        wanted = set(json.loads(Path(args.ids_file).read_text(encoding="utf-8")))
        tasks = [t for t in tasks if t["id"] in wanted]
    if categories:
        tasks = select_categories(tasks, categories)

    records_path = out / "records.jsonl"
    done = load_done(records_path, forced)
    todo = [t for t in tasks if t["id"] not in done]

    print(f"library : {args.library} ({len(library.ids())} skills)")
    if forced:
        print(f"forced  : {', '.join(forced)}")
    if categories:
        print(f"category: {', '.join(categories)}")
    print(f"tasks   : {len(tasks)}  (done {len(done)}, todo {len(todo)})")
    print(f"workers : {workers}")

    q: "queue.Queue" = queue.Queue()
    for t in todo:
        q.put(t)
    lock = threading.Lock()
    t0 = time.time()
    counter = {"n": 0}

    def _worker(worker_id: int) -> None:
        client = client_factory(worker_id)
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            planner_picks: List[str] = []
            message = ""
            final_code = ""
            timing: Dict[str, Any] = {}
            try:
                kwargs: Dict[str, Any] = {}
                if forced is not None:
                    kwargs["forced_skill_ids"] = list(forced)
                rec = run_task_fn(t, client, executor_config=exec_cfg,
                                  library=library, **kwargs)
                prediction, status = rec.prediction, rec.final_status
                message = str(getattr(rec, "final_message", "") or "")[:300]
                chosen = list(rec.chosen_skills or [])
                planner_picks = list(getattr(rec, "planner_skills", None) or [])
                attempts = list(getattr(rec, "code_attempts", None) or [])
                final_code = str(attempts[-1]) if attempts else ""
                timing = dict(getattr(rec, "timing", None) or {})   # per-task cost (usage.py), for RQ5
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[eval w{worker_id}] {type(exc).__name__}: {exc}\n")
                prediction, status, chosen = "", "pipeline_crash", []
                message = f"{type(exc).__name__}: {exc}"[:300]
            try:
                score, _br = score_one(t["eval_metric"], prediction,
                                       t.get("ground_truth", t.get("answer")))
            except Exception:  # noqa: BLE001
                score = 0.0
            row = {
                "task_id": t["id"], "category": t["category"],
                "subtask": t["subtask"], "level": t["level"],
                "eval_metric": t["eval_metric"], "prediction": str(prediction),
                "score": float(score), "status": status,
                "message": message,
                "chosen_skills": chosen,
                "planner_skills": planner_picks,
                "forced_skills": (list(forced) if forced is not None else None),
                # F5: the flags AND the code they were read off, so a later
                # question about the code costs nothing.
                "final_code": final_code,
                "timing": timing,
                **index_usage(final_code),
            }
            with lock:
                with records_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counter["n"] += 1
                if counter["n"] % 10 == 0:
                    print(f"  {counter['n']}/{len(todo)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            q.task_done()

    threads = [threading.Thread(target=_worker, args=(i,), name=f"eval-w{i}")
               for i in range(min(workers, max(1, len(todo))))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    rows = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    wanted_ids = {t["id"] for t in tasks}
    want_forced = forced_key(forced)
    rows = [r for r in rows
            if r["task_id"] in wanted_ids
            and forced_key(r.get("forced_skills")) == want_forced]
    table = build_table(rows)
    table["library"] = args.library
    table["forced_skills"] = list(forced) if forced is not None else None
    table["categories"] = categories
    # D1/D7: the settings the three conditions must share, written next to
    # the numbers they produced.
    table["workers"] = workers
    table["exec_concurrency"] = EXEC_CONCURRENCY
    table["exec_timeout_seconds"] = exec_cfg["timeout_seconds"]
    (out / "table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    title = f"Evaluation — library: {args.library}"
    if forced:
        title += f" — forced: {', '.join(forced)}"
    md = render_markdown(table, title)
    (out / "table.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
