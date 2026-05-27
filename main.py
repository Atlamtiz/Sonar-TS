"""Sonar-TS benchmark runner."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sonar_ts import _paths  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _kaleido_available() -> bool:
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False
    return True


def _worker(thread_id: int, api_key: str, llm_cfg: dict, exec_cfg: dict,
            q: "queue.Queue", partial_lock: threading.Lock,
            partial_path: Path, progress, stats: dict) -> None:
    from sonar_ts.llm import DeepSeekClient
    from sonar_ts.pipeline import run_task

    client = DeepSeekClient(
        api_key=api_key,
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        thinking_mode=llm_cfg.get("thinking_mode", "disabled"),
        max_attempts=llm_cfg["http_max_attempts"],
        backoff_seconds=llm_cfg["http_backoff_seconds"],
    )

    while True:
        try:
            idx, task = q.get_nowait()
        except queue.Empty:
            return
        try:
            rec = run_task(task, client, executor_config=exec_cfg)
            prediction = rec.prediction
            status = rec.final_status
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[t{thread_id}] {task['id']} crash: "
                             f"{type(exc).__name__}: {exc}\n")
            traceback.print_exc(file=sys.stderr)
            prediction = ""
            status = "pipeline_crash"

        with partial_lock:
            with partial_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": idx, "prediction": prediction},
                                   ensure_ascii=False) + "\n")
            stats[status] = stats.get(status, 0) + 1

        if progress is not None:
            progress.update(1)
        q.task_done()


def _read_partial(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    done: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            done[int(entry["id"])] = entry.get("prediction", "")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return done


def _consolidate_predict_json(partial_path: Path, out_path: Path,
                              n_tasks: int) -> dict[int, str]:
    done = _read_partial(partial_path)
    rows = [{"id": idx, "prediction": done.get(idx, "")}
            for idx in range(n_tasks)]
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return done


def _run_pipeline(tasks: list, llm_cfg: dict, exec_cfg: dict, workers: int,
                  partial_path: Path) -> None:
    from sonar_ts.llm import load_api_keys
    keys = load_api_keys(llm_cfg["api_keys_file"])
    if len(keys) < workers:
        raise SystemExit(f"need at least {workers} API keys, "
                         f"found {len(keys)} in {llm_cfg['api_keys_file']}")

    done = _read_partial(partial_path)
    if done:
        print(f"Resuming: {len(done)} tasks already in {partial_path.name}")

    q: queue.Queue = queue.Queue()
    for idx, t in enumerate(tasks):
        if idx in done:
            continue
        q.put((idx, t))
    remaining = q.qsize()
    print(f"Workers: {workers}  (1 dedicated API key each)")
    print(f"Queueing {remaining} tasks (of {len(tasks)} total).\n")
    if remaining == 0:
        return

    progress = tqdm(total=remaining, desc="tasks", ncols=80) if tqdm else None
    partial_lock = threading.Lock()
    stats: dict[str, int] = {}

    threads: list[threading.Thread] = []
    t0 = time.time()
    for i in range(workers):
        th = threading.Thread(
            target=_worker, name=f"sonar-w{i}",
            args=(i, keys[i], llm_cfg, exec_cfg, q,
                  partial_lock, partial_path, progress, stats),
            daemon=False,
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    if progress is not None:
        progress.close()
    elapsed = time.time() - t0

    print(f"\nFinished {remaining} tasks in {elapsed/60:.1f} min")
    if stats:
        print("Status breakdown:")
        for k in sorted(stats):
            print(f"  {k:20s} {stats[k]:>5d}")


def _figure_path(figures_dir: Path, idx: int) -> Path:
    return figures_dir / f"{idx:04d}.png"


_RENDER_TIMEOUT_SECONDS = 30


def _render_timeout_handler(signum, frame):  # noqa: ARG001
    raise TimeoutError(f"render exceeded {_RENDER_TIMEOUT_SECONDS}s")


def _render_one(args: tuple) -> tuple:
    import signal as _signal
    idx, task, prediction, figures_dir_str = args
    from pathlib import Path as _P
    from sonar_ts.postprocess import parse_prediction_for_viz
    from sonar_ts.postprocess.visualize import save_figure

    if hasattr(_signal, "SIGALRM"):
        _signal.signal(_signal.SIGALRM, _render_timeout_handler)
        _signal.alarm(_RENDER_TIMEOUT_SECONDS)
    try:
        result = parse_prediction_for_viz(prediction, task["eval_metric"])
        save_figure(task, result, _P(figures_dir_str) / f"{idx:04d}.png")
        return (idx, True, None)
    except TimeoutError as exc:
        return (idx, False, f"timeout: {exc}")
    except Exception as exc:  # noqa: BLE001
        first = (str(exc).strip().splitlines() or [""])[0][:160]
        return (idx, False, f"{type(exc).__name__}: {first}")
    finally:
        if hasattr(_signal, "SIGALRM"):
            _signal.alarm(0)


def _render_all_figures(tasks: list, predictions: dict[int, str],
                        figures_dir: Path, n_workers: int) -> None:
    # kaleido is not thread-safe; use a process pool so each worker
    # gets its own headless-browser subprocess.
    import multiprocessing as mp

    to_do = [(idx, t, predictions.get(idx, ""), str(figures_dir))
             for idx, t in enumerate(tasks)
             if not _figure_path(figures_dir, idx).exists()]
    if not to_do:
        print(f"\nFigures: all {len(tasks)} PNG(s) already present in "
              f"{figures_dir} — skipping render.")
        return

    n_workers = max(1, min(n_workers, len(to_do)))
    print(f"\nRendering {len(to_do)} figure(s) → {figures_dir} "
          f"({n_workers} parallel processes; kaleido is not thread-safe but"
          f" is fine across processes)…")

    bar = tqdm(total=len(to_do), desc="figures", ncols=80) if tqdm else None
    n_ok = n_err = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for idx, ok, err in pool.imap_unordered(_render_one, to_do,
                                                 chunksize=4):
            if ok:
                n_ok += 1
            else:
                n_err += 1
                sys.stderr.write(f"  viz failed for idx={idx}: {err}\n")
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    print(f"  rendered {n_ok} PNG(s); {n_err} failed.")


def _score_all(tasks: list, predictions: dict[int, str]) -> list[dict]:
    from sonar_ts.evaluator import score_one
    per_row: list[dict] = []
    for idx, t in enumerate(tasks):
        pred = predictions.get(idx, "")
        try:
            score, breakdown = score_one(t["eval_metric"], pred,
                                          t.get("ground_truth", t.get("answer")))
        except Exception:  # noqa: BLE001
            score, breakdown = 0.0, None
        per_row.append({
            "id":          idx,
            "task_id":     t["id"],
            "level":       t["level"],
            "category":    t["category"],
            "subtask":     t["subtask"],
            "eval_metric": t["eval_metric"],
            "prediction":  pred,
            "ground_truth": t.get("answer", t.get("ground_truth")),
            "score":       float(score),
            "breakdown":   breakdown,
        })
    return per_row


def _print_table(agg: dict) -> None:
    from sonar_ts.evaluator import CATEGORY_ORDER, SUBTASK_METRIC
    bar = "=" * 84
    print()
    print(bar)
    print("  NLQTSBench evaluation — Sonar-TS")
    print(bar)

    print("\n  Per-subtask scores (internal labels)")
    print(f"  {'Subtask':24s} {'Metric':10s} {'N':>5s}  {'Score':>8s}")
    print(f"  {'-'*24} {'-'*10} {'-'*5}  {'-'*8}")
    for st, metric in SUBTASK_METRIC.items():
        info = agg["by_subtask"].get(st, {"n": 0, "avg": 0.0})
        print(f"  {st:24s} {metric:10s} {info['n']:>5d}  {info['avg']:>8.4f}")

    print("\n  Per-category scores (paper alignment)")
    print(f"  {'Category':24s} {'Code':6s} {'Level':6s} {'N':>5s}  {'Score':>8s}")
    print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*5}  {'-'*8}")
    for cat, lvl, code in CATEGORY_ORDER:
        info = agg["by_category"].get(cat, {"n": 0, "avg": 0.0})
        print(f"  {cat:24s} {code:6s} {lvl:6s} {info['n']:>5d}  {info['avg']:>8.4f}")

    print("\n  Per-level scores")
    print(f"  {'Level':6s} {'N':>5s}  {'Score':>8s}")
    print(f"  {'-'*6} {'-'*5}  {'-'*8}")
    for lvl in sorted(agg["by_level"]):
        info = agg["by_level"][lvl]
        print(f"  L{lvl:<5d} {info['n']:>5d}  {info['avg']:>8.4f}")

    print()
    print(f"  Overall  N = {agg['overall']['n']:>5d}   "
          f"Score = {agg['overall']['avg']:.4f}")
    print(bar)


def _build_summary(agg: dict) -> dict:
    from sonar_ts.evaluator import CATEGORY_ORDER, SUBTASK_METRIC
    return {
        "overall": agg["overall"],
        "by_level": {f"L{k}": v for k, v in sorted(agg["by_level"].items())},
        "by_category": {
            cat: {
                "code":  code,
                "level": lvl,
                **agg["by_category"].get(cat, {"n": 0, "avg": 0.0}),
            }
            for (cat, lvl, code) in CATEGORY_ORDER
        },
        "by_subtask": {
            st: {
                "metric": SUBTASK_METRIC[st],
                **agg["by_subtask"].get(st, {"n": 0, "avg": 0.0}),
            }
            for st in SUBTASK_METRIC
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(prog="main.py",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=str(_HERE / "nlqtsbench" / "tasks.json"),
                   help="Path to tasks.json (default: ./nlqtsbench/tasks.json).")
    p.add_argument("--databases", default=str(_HERE / "databases"),
                   help="Root directory of per-task SQLite files "
                        "(default: ./databases).")
    p.add_argument("--out-dir", default=str(_HERE / "output"),
                   help="Directory for predictions + reports (default: ./output).")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker thread count (default: from configs/online.yaml).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N tasks (quick smoke test).")
    p.add_argument("--rebuild", action="store_true",
                   help="Drop any existing partial JSONL and start fresh.")
    p.add_argument("--figures", action="store_true",
                   help="Also render one PNG per task (kaleido-based, "
                        "~15-20 min for the full 1153 tasks). Off by "
                        "default so a typical run produces only the "
                        "scoring artefacts.")
    p.add_argument("--figures-dir", default=None,
                   help="Per-task PNG output directory when --figures is "
                        "set (default: <out-dir>/figures).")
    p.add_argument("--figure-workers", type=int, default=None,
                   help="Process count for parallel PNG rendering "
                        "(default: same as --workers, capped at 10).")
    args = p.parse_args()

    _paths.DATABASES_DIR = Path(args.databases).resolve()
    if not _paths.DATABASES_DIR.is_dir():
        raise SystemExit(f"databases root does not exist: {_paths.DATABASES_DIR}")

    tasks_path = Path(args.tasks).resolve()
    if not tasks_path.is_file():
        raise SystemExit(f"tasks.json not found: {tasks_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "predict_partial.jsonl"
    predict_path = out_dir / "predict.json"
    summary_path = out_dir / "summary.json"
    per_task_path = out_dir / "per_task.json"

    figures_dir: Path | None = None
    if args.figures:
        if _kaleido_available():
            figures_dir = (Path(args.figures_dir).resolve() if args.figures_dir
                           else out_dir / "figures")
            figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("Note: --figures was requested but kaleido is not "
                  "installed, so PNG export is disabled for this run.\n"
                  "      To enable visualizations:  pip install -U kaleido")


    with open(_HERE / "configs" / "online.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_cfg = config["llm"]
    exec_cfg = {
        "max_regenerations": config["executor"]["max_regenerations"],
        "timeout_seconds":   config["executor"]["timeout_seconds"],
        "shape_rules":       config["shape_rules"],
    }
    workers = args.workers or config["concurrency"]["workers"]

    tasks = json.load(open(tasks_path, encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Tasks:     {tasks_path}")
    print(f"Databases: {_paths.DATABASES_DIR}")
    print(f"Output:    {out_dir}")
    print(f"Figures:   {figures_dir or '(disabled)'}")
    print(f"N tasks:   {len(tasks)}")

    if args.rebuild and partial_path.exists():
        partial_path.unlink()

    _run_pipeline(tasks, llm_cfg, exec_cfg, workers, partial_path)

    predictions = _consolidate_predict_json(partial_path, predict_path, len(tasks))
    print(f"\nWrote {predict_path}  ({len(predictions)} predictions)")

    if figures_dir is not None:
        fig_workers = args.figure_workers or min(workers, 10)
        _render_all_figures(tasks, predictions, figures_dir, fig_workers)

    from sonar_ts.evaluator import aggregate
    per_row = _score_all(tasks, predictions)
    agg = aggregate(per_row)

    _print_table(agg)
    summary_path.write_text(
        json.dumps(_build_summary(agg), indent=2, ensure_ascii=False),
        encoding="utf-8")
    per_task_path.write_text(
        json.dumps(per_row, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {per_task_path}")


if __name__ == "__main__":
    main()
