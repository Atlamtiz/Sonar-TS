"""Guards against writing into the trees the project declares read-only."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sonar_ts.index_guide import ENV_VAR, SYMBOL_COLUMN_ENV

# `github-code/` and `local-code/` are reference checkouts: never written to.
READONLY_ROOTS = ("github-code", "local-code")


def check_databases_dir(path: Path | str, *,
                        required_ids: Iterable[str] | None = None,
                        max_report: int = 5) -> Path:
    """Resolve a `--databases`/`--train` directory, refusing a read-only tree.

    The executor hands the generated script a plain `sqlite3.connect` (see
    `sonar_ts/_runner.py`), so a stray `UPDATE`/`CREATE TABLE` in generated
    code would modify the database in place — and SQLite writes a journal
    file next to it even for a rolled-back statement.
    """
    resolved = Path(path).resolve()
    for part in resolved.parts:
        if part in READONLY_ROOTS:
            raise SystemExit(
                f"{resolved} lies inside the read-only tree '{part}/'. The "
                f"executor opens the task database read-write, so generated "
                f"code could modify it. Copy the databases first:\n"
                f"    python -m cold_start_stream.make_subset --copy-databases\n"
                f"and point at experiments/E2-skills/eval/databases (or your "
                f"own copy).")
    # A path that does not exist, or that holds none of the databases the run
    # needs, is not a negative result: every execution fails, every score is
    # 0.0 and the report looks well-formed. Say so before the run starts.
    if not resolved.is_dir():
        raise SystemExit(f"--databases {resolved} is not a directory.")
    if required_ids is not None:
        missing: Sequence[str] = [tid for tid in required_ids
                                  if not (resolved / f"{tid}.sqlite").is_file()]
        if missing:
            shown = ", ".join(missing[:max_report])
            more = f" (+{len(missing) - max_report} more)" \
                if len(missing) > max_report else ""
            raise SystemExit(
                f"{len(missing)} of the requested task databases are missing "
                f"from {resolved}: {shown}{more}. Every run on them would fail "
                f"and be stored as score 0.0.")
    return resolved


def check_symbolization_env(env: Mapping[str, str] | None = None) -> None:
    """ADDENDUM H5: a non-SAX arm run against the `sax`-named column.

    `$SONAR_SYMBOLIZATION` picks the codebook; `$SONAR_SYMBOL_COLUMN` picks the
    column the package reads. Nothing ties them together, so setting only the
    first one gives runtime facts that say `sax_len, sax` beside a codebook
    that describes `sym` — and because every failure in `evidence.py` is
    swallowed, the whole run silently carries no INDEX EVIDENCE at all and no
    message. Said out loud at start instead.
    """
    src = os.environ if env is None else env
    symbolization = (src.get(ENV_VAR) or "").strip()
    column = (src.get(SYMBOL_COLUMN_ENV) or "").strip()
    if not symbolization or symbolization.lower() == "sax" or column:
        return
    raise SystemExit(
        f"{ENV_VAR}={symbolization} but {SYMBOL_COLUMN_ENV} is not set, so "
        f"the package would read the column 'sax' while the codebook "
        f"describes another symbolization: every index read fails silently "
        f"and the whole run carries no index evidence. Set "
        f"{SYMBOL_COLUMN_ENV} (e.g. {SYMBOL_COLUMN_ENV}=sym) to the symbolic "
        f"column of these databases, or unset {ENV_VAR} for a SAX run.")
