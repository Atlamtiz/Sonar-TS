"""Does the produced code actually use the symbolic index? (ADDENDUM F5)

Three flags, read off the FINAL code that ran:

    uses_feature_table   a feature table is queried at all
    uses_sax             the `sax` column together with a feature table
    uses_sax_pattern     …and the string is matched with REGEXP / regexp_like /
                         LIKE / GLOB / `re.search|match|fullmatch|finditer|findall`
                         / a pandas `.str.contains|match|fullmatch` call on the
                         symbolic column itself (ADDENDUM H9)

`uses_sax_pattern` is the number the paper's claim rests on: an index search
for the FORM of the curve is a pattern match on `sax`, not a `SELECT sax`
followed by a scan of raw_data. The same three flags are recorded per task by
`eval.py`, per run by `select.py`, and can be recomputed for any record file
that carries code:

    python -m cold_start_stream.index_usage <dir-or-jsonl> [<dir-or-jsonl> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_HERE = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _HERE.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from sonar_ts.index_guide import resolve_symbol_column  # noqa: E402

from .lint import python_blocks  # noqa: E402

FEATURE_TABLE_RE = re.compile(r"\b(?:daily|monthly|yearly)_feature\b")
# E5: there is deliberately no module-level `\bsax\b` constant any more — the
# symbolic column is whatever `resolve_symbol_column` returns, and a hard-coded
# one would silently report "did not use the index" on a `sym` database.


# E5. The symbolic column is not always called `sax`: an arm that puts 1d-SAX,
# TFSAX or MOSO symbols into it renames it (`$SONAR_SYMBOL_COLUMN`). The three
# flags KEEP THEIR NAMES — `uses_sax` is the number the paper reports — but
# they mean "uses the symbolic column", whatever it is called in the databases
# the run was made against. `\b<name>\b`, so `sym_len` still does not count.
@lru_cache(maxsize=None)
def symbol_column_re(column: str) -> "re.Pattern[str]":
    return re.compile(rf"\b{re.escape(column)}\b")


def _column_re(column: str | None) -> "re.Pattern[str]":
    return symbol_column_re(resolve_symbol_column(column))


# SQL keywords are case-insensitive, and generated SQL is written both ways
# (`sax LIKE '%aaa%'` and `sax like '%aaa%'` are the same query). Matching the
# upper-case spelling only would have counted the second one as "did not search
# the index", i.e. it would have understated the one number ADDENDUM F exists
# to move. `LIKE` / `GLOB` / `REGEXP` are also ordinary English words and a
# module name, so they only count when what follows them looks like the
# right-hand side of a SQL operator — a quote, a `?` / `%s` / `:name` binding or
# an f-string hole. That keeps "days that look like a rise" in a comment and
# `glob.glob(...)` out, while `regexp_like(sax, …)` is unambiguous on its own.
_SQL_OPERAND = r"(?=\s*(?:[\"'`]|\?|%|:\w|\{))"
_SQL_PATTERN = (r"(?i:\bregexp_like\b)"
                r"|(?i:\b(?:regexp|like|glob)\b)" + _SQL_OPERAND)
# `re.compile` is here although F5's list names only the matching calls: the
# common idiom is to compile once above the loop and call `.finditer` on the
# compiled object, where no `re.<verb>` is left to see. Counting that as "no
# pattern search" would be a false negative on exactly the intended behaviour.
PATTERN_RE = re.compile(
    _SQL_PATTERN + r"|\bre\.(?:compile|search|match|fullmatch|finditer|findall)\b")
# The skill-level flag of F5 is worded more widely ("an `re.` call"): a skill
# may compile its pattern (`re.compile`) and use it later, which the snippet
# shows as one call.
SKILL_PATTERN_RE = re.compile(_SQL_PATTERN + r"|\bre\.\w+\s*\(")


# ADDENDUM H9. Neither regex above sees the third way of searching the index,
# and it is the one the MOSO round-2 skills actually used: pull the symbolic
# column out with a plain `SELECT` and match it in pandas,
#
#     keep = idx[idx["sym"].str.contains(pat, regex=True, na=False)]
#
# `prominent-upward-spike-interval` and `longest-low-plateau-range` (round 2,
# `runs/moso_r2/s3/library_final`) were both reported as NOT using the
# symbolic index although they search it twice each, and every run-level
# `uses_sax_pattern` rate was understated by the same amount. That is a regex
# search of the index, so it counts.
#
# The match is anchored on the CONFIGURED column (`idx["sym"].str.…` or
# `idx.sym.str.…`), not on any `.str.contains` anywhere: matching a free-text
# `label` column is not an index search, and `uses_sax`'s column test alone
# would not have told the two apart. `.str.match` / `.str.fullmatch` are regex
# calls by definition; `.str.contains` is one unless the call says otherwise,
# so a literal `regex=False` substring test is excluded by the lookahead
# (which does not cross a parenthesis, i.e. it only inspects that call's own
# argument list).
_PANDAS_STR_VERB = r"(?:contains|match|fullmatch)"


@lru_cache(maxsize=None)
def pandas_symbol_pattern_re(column: str) -> "re.Pattern[str]":
    col = re.escape(column)
    access = rf"(?:\[\s*[\"']{col}[\"']\s*\]|\.{col}\b)"
    return re.compile(
        rf"{access}\s*\.\s*str\s*\.\s*{_PANDAS_STR_VERB}\s*\("
        r"(?![^()]*\bregex\s*=\s*False\b)")


def _pandas_re(column: str | None) -> "re.Pattern[str]":
    return pandas_symbol_pattern_re(resolve_symbol_column(column))


# Code that never ran is not evidence of what the code did: a line that is
# entirely a comment ("# the sax column could be used here") is dropped before
# any flag is read, so a mention in prose and a commented-out query both count
# as what they are — nothing. Only whole-line comments are removed, because a
# `#` inside a string literal cannot start one.
_COMMENT_LINE_RE = re.compile(r"^[ \t]*#.*$", re.MULTILINE)

FLAGS = ("uses_feature_table", "uses_sax", "uses_sax_pattern")

# The keys a record can carry its code under: eval.py writes `final_code`, the
# stream's solve records write `code`, and a raw pipeline record keeps the
# attempts.
_CODE_KEYS = ("final_code", "code", "generated_code")


def strip_comment_lines(code: str | None) -> str:
    """Drop whole-line `#` comments; what is left is what actually ran."""
    return _COMMENT_LINE_RE.sub("", str(code or ""))


def index_usage(code: str | None, *, column: str | None = None) -> Dict[str, bool]:
    """The three flags of one script (F5), for the configured column (E5)."""
    text = strip_comment_lines(code)
    table = bool(FEATURE_TABLE_RE.search(text))
    sax = bool(table and _column_re(column).search(text))
    pattern = bool(sax and (PATTERN_RE.search(text)
                            or _pandas_re(column).search(text)))
    return {"uses_feature_table": table, "uses_sax": sax,
            "uses_sax_pattern": pattern}


def skill_uses_symbolic_index(text: str | None, *,
                              column: str | None = None) -> bool:
    """Does a SKILL document's snippet search the symbolic index? (F5)

    A skill counts when its code contains `sax` together with a pattern match;
    a document whose prose merely mentions the column does not. When the
    document has no ```python block the whole text is used, so a bare snippet
    handed in without frontmatter is still judged.

    H9: a pandas `df["sax"].str.contains(pat, regex=True)` counts as the
    pattern match — see `pandas_symbol_pattern_re`.
    """
    doc = str(text or "")
    blocks = python_blocks(doc)
    snippet = strip_comment_lines("\n".join(blocks) if blocks else doc)
    return bool(_column_re(column).search(snippet)
                and (SKILL_PATTERN_RE.search(snippet)
                     or _pandas_re(column).search(snippet)))


# --------------------------------------------------------------------- reading
def code_of(record: Dict[str, Any]) -> str:
    """The final code a record carries, whatever field it lives in."""
    for key in _CODE_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    attempts = record.get("code_attempts")
    if isinstance(attempts, (list, tuple)) and attempts:
        return str(attempts[-1])
    return ""


def _records_of_json(blob: Any) -> List[Dict[str, Any]]:
    """The records inside a whole-file JSON document.

    A stream run keeps its solve records in `state.json` under `records`
    (uid → record), which is the only place the code of an earlier run
    survives; a plain list of records is read too.
    """
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if isinstance(blob, dict):
        inner = blob.get("records", blob)
        if isinstance(inner, dict):
            return [r for r in inner.values() if isinstance(r, dict)]
        if isinstance(inner, list):
            return [r for r in inner if isinstance(r, dict)]
    return []


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Every record of a `.jsonl` (one object per line) or `state.json` file.

    A directory is walked for both kinds.
    """
    files = ([path] if path.is_file()
             else sorted(list(path.rglob("*.jsonl"))
                         + list(path.rglob("state.json"))))
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        if file.suffix == ".json":
            try:
                blob = json.loads(text)
            except json.JSONDecodeError:
                continue
            for row in _records_of_json(blob):
                row.setdefault("_file", str(file))
                yield row
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row.setdefault("_file", str(file))
                yield row


def rates(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The three rates over the records that carry code."""
    # Always recomputed from the code, never read off a stored flag: a record
    # file from an earlier run predates the flags, and mixing the two would
    # make one column mean two different things.
    with_code = [r for r in records if code_of(r).strip()]
    counts = {flag: 0 for flag in FLAGS}
    for r in with_code:
        flags = index_usage(code_of(r))
        for flag in FLAGS:
            counts[flag] += int(flags[flag])
    n = len(with_code)
    return {"n": len(records), "n_with_code": n,
            **{flag: (round(counts[flag] / n, 4) if n else 0.0)
               for flag in FLAGS},
            "counts": counts}


def by_category(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get("category") or "(unknown)"), []).append(r)
    return {cat: rates(rows) for cat, rows in sorted(groups.items())}


def render_table(title: str, overall: Dict[str, Any],
                 per_category: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"## {title}", "",
             "| Category | N | feature table | sax | sax pattern |",
             "|---|---:|---:|---:|---:|"]
    for cat, row in per_category.items():
        lines.append(f"| {cat} | {row['n_with_code']} | "
                     f"{row['uses_feature_table']:.4f} | {row['uses_sax']:.4f} | "
                     f"{row['uses_sax_pattern']:.4f} |")
    lines.append(f"| **all** | {overall['n_with_code']} | "
                 f"{overall['uses_feature_table']:.4f} | "
                 f"{overall['uses_sax']:.4f} | "
                 f"{overall['uses_sax_pattern']:.4f} |")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m cold_start_stream.index_usage", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="record files (*.jsonl) or directories holding them.")
    p.add_argument("--json", action="store_true",
                   help="print the numbers as JSON instead of a table.")
    args = p.parse_args(argv)

    out: Dict[str, Any] = {}
    for raw in args.paths:
        path = Path(raw)
        records = list(iter_records(path))
        overall = rates(records)
        per_cat = by_category(records)
        out[str(path)] = {"overall": overall, "by_category": per_cat}
        if not args.json:
            print(render_table(str(path), overall, per_cat))
            print()
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
