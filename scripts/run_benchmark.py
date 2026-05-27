"""Run the pipeline over tasks.json with worker threads + JSONL checkpointing."""

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sonar_ts._paths import PROJECT_ROOT, TASKS_JSON
from sonar_ts.llm import DeepSeekClient, load_api_keys
from sonar_ts.pipeline import run_task
from sonar_ts.postprocess.visualize import save_figure

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "online.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_client(api_key: str, llm_cfg: dict) -> DeepSeekClient:
    return DeepSeekClient(
        api_key=api_key,
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        thinking_mode=llm_cfg.get("thinking_mode", "disabled"),
        max_attempts=llm_cfg["http_max_attempts"],
        backoff_seconds=llm_cfg["http_backoff_seconds"],
    )


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


def _consolidate(partial_path: Path, out_path: Path, n_tasks: int) -> None:
    done = _read_partial(partial_path)
    rows = [{"id": idx, "prediction": done.get(idx, "")} for idx in range(n_tasks)]
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def _worker(thread_id: int, api_key: str, llm_cfg: dict,
            exec_cfg: dict, q: queue.Queue,
            partial_lock: threading.Lock, partial_path: Path,
            figures_dir: Path | None, progress, stats: dict) -> None:
    client = _build_client(api_key, llm_cfg)
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return
        idx, task = item
        rec = None
        try:
            rec = run_task(task, client, executor_config=exec_cfg)
            prediction = rec.prediction
            status = rec.final_status
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[thread {thread_id}] {task['id']} pipeline crash: "
                             f"{type(exc).__name__}: {exc}\n")
            traceback.print_exc(file=sys.stderr)
            prediction = ""
            status = "pipeline_crash"

        if figures_dir is not None:
            try:
                save_figure(task, rec.raw_result if rec is not None else None,
                            figures_dir / f"{task['id']}.png")
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[thread {thread_id}] {task['id']} viz failed: "
                                 f"{type(exc).__name__}: {exc}\n")

        with partial_lock:
            with partial_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": idx, "prediction": prediction},
                                   ensure_ascii=False) + "\n")
            stats[status] = stats.get(status, 0) + 1

        if progress is not None:
            progress.update(1)
        q.task_done()


def main() -> None:
    p = argparse.ArgumentParser(prog="run_benchmark",
                                description="Run the Sonar-TS pipeline over all tasks.")
    p.add_argument("--out", default=None,
                   help="Output predict.json path (default: predict.json under project root).")
    p.add_argument("--partial", default=None,
                   help="Incremental JSONL path (default: predict_partial.jsonl under project root).")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker thread count (default: from configs/online.yaml).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N tasks (for testing).")
    p.add_argument("--subtask", default=None,
                   help="Only run tasks whose subtask field equals this value "
                        "(e.g. 'Sliding Window'). Combine with --rebuild for "
                        "clean per-subtask iteration.")
    p.add_argument("--rebuild", action="store_true",
                   help="Discard any existing partial file and start fresh.")
    p.add_argument("--viz", action="store_true",
                   help="Also save one PNG per task to output/figures/.")
    p.add_argument("--figures-dir", default=None,
                   help="Override the PNG output directory (default: output/figures/).")
    args = p.parse_args()

    config = _load_config()
    llm_cfg = config["llm"]
    exec_cfg = {
        "max_regenerations": config["executor"]["max_regenerations"],
        "timeout_seconds": config["executor"]["timeout_seconds"],
        "shape_rules": config["shape_rules"],
    }
    workers = args.workers or config["concurrency"]["workers"]
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else output_dir / "predict.json"
    partial_path = Path(args.partial) if args.partial else output_dir / "predict_partial.jsonl"

    keys = load_api_keys(llm_cfg["api_keys_file"])
    if len(keys) < workers:
        raise SystemExit(f"need at least {workers} API keys, found {len(keys)}")
    print(f"Workers: {workers}  (1 dedicated API key each)")
    print(f"Out:     {out_path}")
    print(f"Partial: {partial_path}")

    if args.rebuild and partial_path.exists():
        partial_path.unlink()

    figures_dir: Path | None = None
    if args.viz:
        figures_dir = Path(args.figures_dir) if args.figures_dir \
            else output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"Figures: {figures_dir}")

    all_tasks = json.load(open(TASKS_JSON, encoding="utf-8"))
    n_total = len(all_tasks)
    # Preserve ORIGINAL tasks.json positions as the `id` in predict.json so the
    # evaluator can join filtered results back against the full gold.
    indexed_tasks = list(enumerate(all_tasks))
    if args.subtask:
        before = len(indexed_tasks)
        indexed_tasks = [(i, t) for i, t in indexed_tasks if t.get("subtask") == args.subtask]
        if not indexed_tasks:
            raise SystemExit(f"no tasks match --subtask {args.subtask!r}")
        print(f"Filtered to subtask {args.subtask!r}: {len(indexed_tasks)} of {before} tasks")
    if args.limit:
        indexed_tasks = indexed_tasks[: args.limit]



    done = _read_partial(partial_path)
    if done:
        print(f"Resuming: {len(done)} tasks already in {partial_path.name}")

    q: queue.Queue = queue.Queue()
    for idx, t in indexed_tasks:
        if idx in done:
            continue
        q.put((idx, t))
    remaining = q.qsize()
    print(f"Queueing {remaining} tasks (of {len(indexed_tasks)} in scope).\n")
    if remaining == 0:
        _consolidate(partial_path, out_path, n_total)
        print(f"Already complete — wrote {out_path}")
        return

    progress = tqdm(total=remaining, desc="tasks") if tqdm else None
    partial_lock = threading.Lock()
    stats: dict[str, int] = {}

    threads: list[threading.Thread] = []
    t0 = time.time()
    for i in range(workers):
        th = threading.Thread(
            target=_worker, name=f"sonar-w{i}",
            args=(i, keys[i], llm_cfg, exec_cfg, q,
                  partial_lock, partial_path, figures_dir, progress, stats),
            daemon=False,
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    if progress is not None:
        progress.close()
    elapsed = time.time() - t0

    _consolidate(partial_path, out_path, n_total)

    print(f"\nFinished {remaining} tasks in {elapsed/60:.1f} min")
    print(f"Status breakdown:")
    for k in sorted(stats):
        print(f"  {k:20s} {stats[k]:>5d}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
