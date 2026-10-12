"""Build the held-out evaluation subset: 30 tasks per sub-task, fixed seed.

    python -m cold_start_stream.make_subset [--tasks …] [--out …]
        [--per-subtask 30] [--seed 20260920] [--copy-databases <src dir>]

The executor opens the task database with a plain `sqlite3.connect` (not
`file:…?mode=ro`), so the read-only source databases must be copied before an
evaluation run; `--copy-databases` does exactly that.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
_CODE_ROOT = _FRAMEWORK_ROOT.parent
_E2 = _CODE_ROOT / "experiments" / "E2-skills"
_GITHUB_CODE = _CODE_ROOT.parent.parent / "github-code" / "Sonar-TS"

DEFAULT_TASKS = _GITHUB_CODE / "nlqtsbench" / "tasks.json"
DEFAULT_DBS = _GITHUB_CODE / "databases"
DEFAULT_OUT = _E2 / "eval" / "subset_330.json"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m cold_start_stream.make_subset",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=str(DEFAULT_TASKS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--per-subtask", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--copy-databases", nargs="?", const=str(DEFAULT_DBS),
                   default=None,
                   help="Copy the subset's databases out of this read-only "
                        "source directory into <out dir>/databases.")
    args = p.parse_args(argv)

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    by_subtask: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_subtask[t["subtask"]].append(t)

    rng = random.Random(args.seed)
    picked: list[str] = []
    counts: dict[str, int] = {}
    for subtask in sorted(by_subtask):
        pool = sorted(by_subtask[subtask], key=lambda t: t["id"])
        k = min(args.per_subtask, len(pool))
        chosen = rng.sample(pool, k)
        counts[subtask] = len(chosen)
        picked.extend(t["id"] for t in chosen)
    picked.sort()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=2), encoding="utf-8")
    print(f"{len(picked)} ids from {len(by_subtask)} sub-tasks → {out}")
    for subtask, n in sorted(counts.items()):
        print(f"  {subtask:24s} {n}")

    if args.copy_databases:
        src = Path(args.copy_databases).resolve()
        dst = out.parent / "databases"
        dst.mkdir(parents=True, exist_ok=True)
        n_copied, missing = 0, []
        for task_id in picked:
            s = src / f"{task_id}.sqlite"
            d = dst / f"{task_id}.sqlite"
            if not s.is_file():
                missing.append(task_id)
                continue
            if not d.is_file():
                shutil.copy2(s, d)
            n_copied += 1
        print(f"databases: {n_copied} present in {dst}"
              + (f"; MISSING {len(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
