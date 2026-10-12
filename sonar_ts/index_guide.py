"""The INDEX GUIDE: shared index text plus the codebook of a symbolization.

ADDENDUM F1. Every symbolization ships with a CODEBOOK — the usage manual of
its symbolic column — and the guide the planner and the code generator see is
always the same shared text with that codebook pasted into it:

    index_guide = index_common.replace("<<CODEBOOK>>", codebook.rstrip())

`sonar_ts/prompts/` is the ONE source of truth for both halves: the distiller
(`cold_start_stream`) composes its runtime facts from exactly these files, so
the online pipeline and the distilled skills cannot drift apart.

The guide contains literal braces (`a{n,}`), so it may only ever be passed to
`str.format` as a VALUE, never be part of a template.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from ._paths import PROJECT_ROOT

PROMPTS_DIR: Path = PROJECT_ROOT / "sonar_ts" / "prompts"
COMMON_FILE: str = "index_common.txt"
CODEBOOK_MARKER: str = "<<CODEBOOK>>"
DEFAULT_SYMBOLIZATION: str = "moso"
ENV_VAR: str = "SONAR_SYMBOLIZATION"

# E5. The symbolic column is called `sax` in the databases the pipeline was
# built on, but an experiment that puts 1d-SAX, TFSAX or MOSO symbols into it
# renames it, so that a column called `sax` never holds non-SAX symbols. The
# name is a deployment setting, not a code constant: the three online templates
# carry it as the format field `{symcol}` and the databases are built with the
# same name (see E5-codebook-e2e/build_dbs.py).
SYMBOL_COLUMN_ENV: str = "SONAR_SYMBOL_COLUMN"
DEFAULT_SYMBOL_COLUMN: str = "sym"
GUIDE_DIR_ENV: str = "SONAR_INDEX_GUIDE_DIR"
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def resolve_symbol_column(column: str | None = None) -> str:
    """Explicit argument, else `$SONAR_SYMBOL_COLUMN`, else `sym`.

    The value is pasted into prompts and used to build a word regex, so a name
    that is not a bare SQL identifier is refused instead of silently producing
    a broken prompt or a regex that matches everything.
    """
    raw = (column if column is not None
           else os.environ.get(SYMBOL_COLUMN_ENV, "")) or ""
    name = str(raw).strip()
    if not name:
        return DEFAULT_SYMBOL_COLUMN
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"{SYMBOL_COLUMN_ENV}={name!r} is not a column name: it must be "
            f"letters, digits and underscores and must not start with a digit. "
            f"It is pasted into every prompt and into the index-usage regex.")
    return name.lower()


def resolve_prompts_dir(prompts_dir: Path | str | None = None) -> Path:
    """Explicit argument, else `$SONAR_INDEX_GUIDE_DIR`, else `sonar_ts/prompts`."""
    if prompts_dir is not None:
        return Path(prompts_dir)
    env = os.environ.get(GUIDE_DIR_ENV, "").strip()
    return Path(env) if env else PROMPTS_DIR


def codebook_file(symbolization: str) -> str:
    """The file name of a symbolization's codebook (`codebook_sax.txt`)."""
    return f"codebook_{symbolization}.txt"


def resolve_symbolization(symbolization: str | None = None) -> str:
    """Explicit argument, else `$SONAR_SYMBOLIZATION`, else `moso`."""
    if symbolization:
        return str(symbolization).strip().lower()
    env = os.environ.get(ENV_VAR, "").strip().lower()
    return env or DEFAULT_SYMBOLIZATION


def _read(path: Path, what: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"the index guide needs {what} at {path}, which does not exist. "
            f"Copy it from the design prompts (ADDENDUM F1); without it the "
            f"planner and the code generator cannot be rendered.")
    return path.read_text(encoding="utf-8")


def compose_index_guide(common: str, codebook: str) -> str:
    """Paste a codebook into the shared text at the `<<CODEBOOK>>` marker."""
    if CODEBOOK_MARKER not in common:
        raise ValueError(
            f"the shared index text does not contain the marker "
            f"{CODEBOOK_MARKER}; the codebook would silently be dropped and "
            f"every prompt would ship a guide without its usage manual.")
    return common.replace(CODEBOOK_MARKER, codebook.rstrip())


@lru_cache(maxsize=None)
def _load(symbolization: str, prompts_dir: str) -> str:
    base = Path(prompts_dir)
    common = _read(base / COMMON_FILE, "the shared index text")
    codebook = _read(base / codebook_file(symbolization),
                     f"the codebook of symbolization {symbolization!r}")
    return compose_index_guide(common, codebook)


def load_index_guide(symbolization: str | None = None, *,
                     prompts_dir: Path | str | None = None) -> str:
    """The composed guide for `symbolization` (default: `sax`), cached.

    Raises `FileNotFoundError` when a file is missing and `ValueError` when the
    `<<CODEBOOK>>` marker is gone — both are silent disasters otherwise: the
    guide is a prompt value, so a half-built one simply produces worse answers.
    """
    return _load(resolve_symbolization(symbolization),
                 str(resolve_prompts_dir(prompts_dir)))


def clear_cache() -> None:
    """Forget the cached guides (tests that write their own prompt files)."""
    _load.cache_clear()
