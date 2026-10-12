"""CLI: cut the training log into a TRAIN and a held-out VALID half (E1).

    python -m cold_start_stream.make_split --train <dir with tasks.json>
        --out <runs/full_v4/split.json> [--n-train 300] [--seed 20260921]

The split must be label-free: it is a seeded random permutation of the items
that `load_stream` keeps (the same sanity filter the stream applies), and the
first `--n-train` of them are TRAIN. Nothing here looks at `subtask`,
`category` or `level` — those are only counted afterwards, by `--counts`, as
analysis of a split that was already fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from .stream import load_stream  # noqa: E402

DEFAULT_SEED = 20260921
DEFAULT_N_TRAIN = 300


def build_split(tasks_json: Path | str, *, seed: int = DEFAULT_SEED,
                n_train: int = DEFAULT_N_TRAIN,
                stats: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """`{"seed", "n_train", "n_valid", "train": [ids], "valid": [ids]}`.

    `load_stream` already shuffles with the given seed, so its first `n_train`
    items are the TRAIN half of the permutation E1 asks for.
    """
    load_stats: Dict[str, Any] = {}
    items = load_stream(tasks_json, seed=seed, stats=load_stats)
    if stats is not None:
        stats.update(load_stats)
    if n_train >= len(items):
        raise SystemExit(f"--n-train {n_train} leaves no VALID items "
                         f"({len(items)} scorable items)")
    ids = [it.task_id for it in items]
    train, valid = ids[:n_train], ids[n_train:]
    return {"seed": seed, "n_train": len(train), "n_valid": len(valid),
            "train": train, "valid": valid}


def per_subtask_counts(tasks_json: Path | str,
                       split: Dict[str, Any]) -> Dict[str, Any]:
    """ANALYSIS ONLY — the labels the split itself was not allowed to see."""
    tasks = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    by_id = {str(t["id"]): t for t in tasks}

    def _count(ids: List[str], field: str) -> Dict[str, int]:
        c = Counter(str(by_id[i].get(field, "")) for i in ids if i in by_id)
        return dict(sorted(c.items()))

    return {"train": {"subtask": _count(split["train"], "subtask"),
                      "category": _count(split["train"], "category")},
            "valid": {"subtask": _count(split["valid"], "subtask"),
                      "category": _count(split["valid"], "category")}}


def render_counts(counts: Dict[str, Any]) -> str:
    """A table of the per-sub-task counts of both halves."""
    names = sorted(set(counts["train"]["subtask"]) | set(counts["valid"]["subtask"]))
    lines = ["| Sub-task | TRAIN | VALID | total |", "|---|---:|---:|---:|"]
    tot_tr = tot_va = 0
    for name in names:
        tr = counts["train"]["subtask"].get(name, 0)
        va = counts["valid"]["subtask"].get(name, 0)
        tot_tr += tr
        tot_va += va
        lines.append(f"| {name} | {tr} | {va} | {tr + va} |")
    lines.append(f"| **total** | **{tot_tr}** | **{tot_va}** | "
                 f"**{tot_tr + tot_va}** |")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m cold_start_stream.make_split",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True,
                   help="training directory with tasks.json (or the file).")
    p.add_argument("--out", required=True, help="split.json to write.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    p.add_argument("--counts", action="store_true",
                   help="print the per-sub-task counts (analysis only).")
    args = p.parse_args(argv)

    train = Path(args.train).resolve()
    tasks_json = train if train.is_file() else train / "tasks.json"
    if not tasks_json.is_file():
        raise SystemExit(f"no tasks.json in {train}")

    stats: Dict[str, Any] = {}
    split = build_split(tasks_json, seed=args.seed, n_train=args.n_train,
                        stats=stats)
    split["tasks_json"] = str(tasks_json)
    split["loader"] = stats
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    # `--ids-file` (run.py) and `--ids-file` (eval.py / select.py) take a plain
    # JSON list, so each half is also written on its own.
    for half in ("train", "valid"):
        (out.parent / f"{half}_ids.json").write_text(
            json.dumps(split[half], indent=2), encoding="utf-8")

    print(f"tasks   : {stats.get('total')} ({stats.get('kept')} scorable, "
          f"{stats.get('skipped')} skipped {stats.get('skipped_by_reason')})")
    print(f"split   : {split['n_train']} TRAIN / {split['n_valid']} VALID "
          f"(seed {split['seed']}) → {out}")
    if args.counts:
        print()
        print(render_counts(per_subtask_counts(tasks_json, split)))


if __name__ == "__main__":
    main()
