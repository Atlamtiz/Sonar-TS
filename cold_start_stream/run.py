"""CLI: distil skills from a shuffled, unlabeled stream of training questions.

    python -m cold_start_stream.run --train <dir> --out <run dir> [--limit N]
        [--batch-size B] [--resume] [--workers W] [--ids-file <json>]
        [--seed S]

`--ids-file` (ADDENDUM E2) restricts the stream to the benchmark task ids in a
JSON list — the TRAIN half of `split.json`; `--seed` (a `StreamConfig` field,
so it was already a flag) is the stream's shuffle seed, which the three
independent streams vary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from sonar_ts import _paths as paths_mod  # noqa: E402
from sonar_ts.llm import DeepSeekClient, load_api_keys  # noqa: E402

from .config import StreamConfig  # noqa: E402
from .guards import (check_databases_dir,  # noqa: E402
                     check_symbolization_env)
from .orchestrator import StreamOrchestrator  # noqa: E402
from .stream import channel_literals, load_stream  # noqa: E402

_CONFIG_PATH = _FRAMEWORK_ROOT / "configs" / "online.yaml"


def wire_train_dir(train: Path) -> Path:
    """Point sonar_ts at the training data (same trick as run_cold_start.py)."""
    tasks_json = train / "tasks.json"
    if not tasks_json.is_file():
        raise SystemExit(f"no tasks.json in {train}")
    paths_mod.TASKS_JSON = tasks_json
    # The stream executes generated code against these databases: refuse a
    # read-only checkout (copy the DBs you need into the experiment dir).
    paths_mod.DATABASES_DIR = check_databases_dir(train / "databases")
    paths_mod.TS_DATA_DIR = train / "ts_data"
    return tasks_json


def resolve_workers(requested: int, n_keys: int) -> int:
    """Worker count to run with, given `n_keys` API keys.

    `client_factory` hands out keys round-robin, so more workers than keys is
    supported: it means ceil(workers / n_keys) concurrent requests per key.
    Only a non-positive request is corrected; a missing key file is fatal.
    """
    if n_keys <= 0:
        raise SystemExit("no API keys loaded")
    return max(1, requested)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cold_start_stream.run",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True,
                   help="Directory with tasks.json, databases/, ts_data/.")
    p.add_argument("--out", required=True, help="Run directory.")
    p.add_argument("--limit", type=int, default=0,
                   help="Use only the first N items of the shuffled stream.")
    p.add_argument("--ids-file", default=None,
                   help="JSON list of benchmark task ids the stream is "
                        "restricted to (E2: the TRAIN half of split.json).")
    p.add_argument("--resume", action="store_true",
                   help="Continue from <out>/state.json.")
    p.add_argument("--config", default=str(_CONFIG_PATH),
                   help="online.yaml to take llm/executor settings from.")
    defaults = StreamConfig()
    for f in fields(StreamConfig):
        flag = "--" + f.name.replace("_", "-")
        p.add_argument(flag, type=type(getattr(defaults, f.name)),
                       default=None, dest=f.name,
                       help=f"(default: {getattr(defaults, f.name)})")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # ADDENDUM H5: a non-SAX symbolization against the default column
    # name loses every index read, silently. Abort before the run.
    check_symbolization_env()

    cfg = StreamConfig()
    for f in fields(StreamConfig):
        value = getattr(args, f.name)
        if value is not None:
            setattr(cfg, f.name, value)

    train = Path(args.train).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks_json = wire_train_dir(train)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    llm_cfg = config["llm"]
    exec_cfg = {
        "max_regenerations": config["executor"]["max_regenerations"],
        "timeout_seconds": config["executor"]["timeout_seconds"],
        "shape_rules": config["shape_rules"],
    }
    api_keys = load_api_keys(llm_cfg["api_keys_file"])
    cfg.workers = resolve_workers(cfg.workers, len(api_keys))

    def client_factory(worker_id: int) -> DeepSeekClient:
        return DeepSeekClient(
            api_key=api_keys[worker_id % len(api_keys)],
            base_url=llm_cfg["base_url"], model=llm_cfg["model"],
            temperature=llm_cfg["temperature"], max_tokens=llm_cfg["max_tokens"],
            thinking_mode=llm_cfg.get("thinking_mode", "disabled"),
            max_attempts=llm_cfg["http_max_attempts"],
            backoff_seconds=llm_cfg["http_backoff_seconds"],
        )

    ids = None
    if args.ids_file:
        ids = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
        if not isinstance(ids, list) or not ids:
            raise SystemExit(f"--ids-file {args.ids_file} is not a non-empty "
                             f"JSON list of task ids")

    load_stats: dict = {}
    items = load_stream(tasks_json, seed=cfg.seed, limit=args.limit,
                        stats=load_stats, ids=ids)
    if not items:
        raise SystemExit("the stream is empty (check --ids-file / --train)")

    print("=" * 78)
    print("  Sonar-TS streaming cold start")
    print("=" * 78)
    print(f"  Train      : {train}")
    print(f"  Out        : {out}")
    print(f"  Stream     : {len(items)} items, batch {cfg.batch_size}, "
          f"{cfg.epochs} epoch(s), shuffle seed {cfg.seed}")
    if ids is not None:
        missing = load_stats.get("ids_not_found") or []
        print(f"  Ids file   : {args.ids_file} "
              f"({load_stats.get('ids_requested', 0)} ids, "
              f"{load_stats.get('selected', 0)} found"
              + (f", {len(missing)} MISSING" if missing else "") + ")")
    print(f"  Sanity     : {load_stats.get('skipped', 0)} of "
          f"{load_stats.get('total', 0)} tasks unscorable and skipped "
          f"{load_stats.get('skipped_by_reason', {})}")
    print(f"  Model      : {llm_cfg['model']} "
          f"(thinking {llm_cfg.get('thinking_mode', 'disabled')})")
    print(f"  Workers    : {cfg.workers} over {len(api_keys)} keys "
          f"({-(-cfg.workers // len(api_keys))} concurrent per key)")
    print(f"  Resume     : {args.resume}")
    print()

    orch = StreamOrchestrator(
        config=cfg, items=items, out_dir=out, exec_cfg=exec_cfg,
        client_factory=client_factory,
        channel_literals=channel_literals(items),
    )
    record = json.dumps({"config": cfg.to_dict(), "train": str(train),
                         "limit": args.limit, "model": llm_cfg["model"],
                         "ids_file": args.ids_file, "resume": args.resume,
                         "loader": load_stats}, indent=2)
    # config.json is the provenance of the invocation that STARTED the run: a
    # resume with different flags must not overwrite it. Each resume writes its
    # own file next to it instead.
    target = out / "config.json"
    if args.resume and target.is_file():
        n = 1
        while (out / f"config.resume{n}.json").is_file():
            n += 1
        target = out / f"config.resume{n}.json"
    target.write_text(record, encoding="utf-8")
    summary = orch.run(resume=args.resume)

    print("\n" + "=" * 78)
    print("  Finished")
    print("=" * 78)
    print(summary["manifest"])
    print(json.dumps(summary["counters"], indent=2))
    print(f"\nRun dir: {out}")


if __name__ == "__main__":
    main()
