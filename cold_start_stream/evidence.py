"""INDEX EVIDENCE for the reflector (ADDENDUM F3).

The reflector used to see the question, the expected answer, the produced
answer and the code — but never what the INDEX looks like where the answer is.
It could therefore say "the pattern was too strict" only as a guess. This
module reads, per failed case, the feature-table rows at the correct answer and
at the produced answer and renders them with the letters of the window that
cover the answer's time span in brackets:

    expected  2019-03-05 00:00:00 → 2019-03-08 00:00:00
      monthly_feature 2019-03-01  slope=+0.443 std=0.139 range=0.410  sax=aaaaaaaa[aabbccdd]ddeeeeeeeeeeeed   (1 letter = 1 day)

ADDENDUM G4: a question that names an upstream and a downstream series is
about both of them, so up to two channel ids are shown, each block labelled
with its channel; with one channel the block is exactly the one F3 rendered.

E5: the symbolic column is not always called `sax` and one "letter" is not
always one character. The column name comes from `$SONAR_SYMBOL_COLUMN`
(resolved exactly as the online prompts and `index_usage` resolve it) and the
width of one TOKEN is read off the row itself — `len(sym) // sym_len`, i.e. 1
for SAX, 2 for 1d-SAX / TFSAX and 4 for MOSO. Brackets always enclose WHOLE
tokens: token i occupies the characters [w*i, w*(i+1)).

Everything here is best effort: the database is opened READ-ONLY and any
failure (no database, no channel, an unparsable answer) yields an empty string,
because a missing piece of evidence must never break a reflection.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import quote

from sonar_ts.evaluator import parse_interval, parse_timestamp
from sonar_ts.index_guide import resolve_symbol_column
from sonar_ts.storage import FEATURE_TABLES

MAX_CHARS = 1400
MAX_SPANS_PER_SIDE = 3
MAX_ROWS_PER_SPAN = 2
# ADDENDUM G4: a question that names an upstream and a downstream series is
# about BOTH of them, and F3's "the first channel id in question order" then
# showed the wrong one (all 45 causal-anomaly tasks of `train_v0` read "given
# that channel X is the upstream source of channel Y, … where Y shows …").
# Up to two ids are used; the second one costs the space of the first one's
# second row, so the block stays inside the same cap.
MAX_CHANNELS = 2
MAX_ROWS_PER_SPAN_MULTI = 1

# Which table resolves a span of this length (F3): the letters must be short
# enough to show the shape and long enough to cover it.
DAILY_MAX = timedelta(hours=36)
MONTHLY_MAX = timedelta(days=45)

# The fallback annotation per table, used only when a row's own window cannot
# be read. The real annotation is computed per row by `letter_unit`: the
# feature tables do NOT always hold one letter per hour / day / month (in
# `train_v0` a few hundred rows carry a shorter `sax_len` than the window has
# hours or days, and one yearly row carries a single letter), and telling the
# reflector "1 letter = 1 hour" over a row where a letter is 80 minutes is a
# false fact in the one block that exists to teach it how to read the index.
LETTER_UNIT = {
    "daily_feature": "1 letter = 1 hour",
    "monthly_feature": "1 letter = 1 day",
    "yearly_feature": "1 letter = 1 month",
}

# (name, seconds, relative tolerance). A calendar month is 28–31 days, so its
# tolerance has to be wide; the others are exact in the data and stay tight.
_UNIT_STEPS = (
    ("1 hour", 3600.0, 0.02),
    ("1 day", 86400.0, 0.02),
    ("1 month", 2629800.0, 0.12),
    ("1 year", 31557600.0, 0.02),
)

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_MIDNIGHT = " 00:00:00"


def token_width(text: Any, n_tokens: Any) -> int:
    """How many characters one token of this row's symbolic string takes (E5).

    Read off the row (`len(sym) // sym_len`) rather than off a setting, because
    the width is a property of the string that is actually stored: 1 for SAX,
    2 for 1d-SAX / TFSAX, 4 for MOSO. Anything that does not divide evenly
    (a truncated or absent `sym_len`) falls back to 1, which is what the code
    did before E5.
    """
    body = str(text or "")
    try:
        n = int(n_tokens or 0)
    except (TypeError, ValueError):
        return 1
    if n <= 0 or not body or len(body) % n:
        return 1
    return max(1, len(body) // n)


def _unit_word(width: int) -> str:
    """`letter` for one-character symbols, `token (w chars)` otherwise."""
    return "letter" if width <= 1 else f"token ({width} chars)"


@dataclass(frozen=True)
class Span:
    """A time span of an answer, with the side/kind it came from."""

    label: str                      # expected | produced | missed | wrong
    start: datetime
    end: datetime

    @property
    def is_instant(self) -> bool:
        return self.end <= self.start


# --------------------------------------------------------------------- channel
def _word_pattern(channel_id: str) -> re.Pattern[str]:
    """Case-sensitive, whole-token match of a channel id in prose."""
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(channel_id)}"
                      rf"(?![A-Za-z0-9_])")


def pick_channels(question: str, channel_ids: Iterable[str],
                  max_n: int = MAX_CHANNELS) -> List[str]:
    """The channel ids the question names, in question order (ADDENDUM G4).

    Longest ids first, so that `OT_2` wins over `OT` where both would match at
    the same place; the result is ordered by the position of the first mention
    and cut to `max_n`, because a question that names an upstream and a
    downstream series is about both of them.
    """
    text = str(question or "")
    hits: List[Tuple[int, int, str]] = []
    taken: List[Tuple[int, int]] = []
    for cid in sorted({str(c) for c in channel_ids if str(c).strip()},
                      key=lambda c: (-len(c), c)):
        # A shorter id must not be "found" inside the span of a longer one it
        # is a part of (`OT` inside `OT_2` is one mention, not two).
        for m in _word_pattern(cid).finditer(text):
            if any(lo <= m.start() and m.end() <= hi for lo, hi in taken):
                continue
            hits.append((m.start(), -len(cid), cid))
            taken.append((m.start(), m.end()))
            break
    return [cid for _s, _n, cid in sorted(hits)][:max(0, int(max_n))]


def pick_channel(question: str, channel_ids: Iterable[str]) -> str | None:
    """The first channel id of the database that the question names."""
    picked = pick_channels(question, channel_ids, max_n=1)
    return picked[0] if picked else None


def channel_ids_of(conn: sqlite3.Connection) -> List[str]:
    """Every `channel_id` that occurs in one of the three feature tables."""
    out: List[str] = []
    seen: set[str] = set()
    for table in FEATURE_TABLES:
        try:
            rows = conn.execute(
                f"SELECT DISTINCT channel_id FROM {table}").fetchall()
        except sqlite3.Error:
            continue
        for row in rows:
            cid = str(row[0])
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


# ----------------------------------------------------------------------- spans
def _instant(label: str, value: Any) -> List[Span]:
    ts = parse_timestamp(value)
    if ts is None:
        return []
    at = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return [Span(label, at, at)]


def _interval(label: str, value: Any) -> List[Span]:
    iv = parse_interval(value)
    if iv is None:
        return []
    start, end = iv
    a = start.to_pydatetime() if hasattr(start, "to_pydatetime") else start
    b = end.to_pydatetime() if hasattr(end, "to_pydatetime") else end
    return [Span(label, a, b)]


def _day_spans(label: str, dates: Sequence[str]) -> List[Span]:
    spans: List[Span] = []
    for day in dates:
        try:
            start = datetime.strptime(str(day)[:10], "%Y-%m-%d")
        except ValueError:
            continue
        spans.append(Span(label, start, start + timedelta(days=1)))
    return spans


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _ordered_dates(value: Any) -> List[str]:
    """The dates of a `set_f1` value in the order the answer lists them.

    `parse_date_set` returns a set, so the three dates shown as evidence would
    be the three EARLIEST ones. The answer is a ranked list: the three the
    method put FIRST are the ones that say what it got wrong, so the order of
    the list is kept here and duplicates are dropped by first occurrence.
    """
    if isinstance(value, (list, tuple, set)):
        flat: List[Any] = []
        for v in value:
            if isinstance(v, (list, tuple, set)):
                flat.extend(v)
            else:
                flat.append(v)
        found = [str(v)[:10] for v in flat if _DATE_RE.match(str(v)[:10])]
    else:
        found = _DATE_RE.findall(str(value))
    out: List[str] = []
    seen: set[str] = set()
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def spans_of(eval_metric: str, expected: Any, produced: Any) -> List[Span]:
    """The spans of one case, by eval_metric (F3).

    `hit` is one timestamp per side, `iou` one interval per side, `set_f1` up
    to three dates the answer MISSED and up to three it produced WRONGLY —
    the dates that differ are the ones that say what the method got wrong.
    Those are the FIRST three in the order of the answer list (rank order),
    not the three earliest dates.
    `rel_acc` and `report` have no span, so they get no evidence. An
    unparsable value drops that side only.
    """
    metric = str(eval_metric or "").strip().lower()
    if metric == "hit":
        return (_instant("expected", expected)[:MAX_SPANS_PER_SIDE]
                + _instant("produced", produced)[:MAX_SPANS_PER_SIDE])
    if metric == "iou":
        return (_interval("expected", expected)[:MAX_SPANS_PER_SIDE]
                + _interval("produced", produced)[:MAX_SPANS_PER_SIDE])
    if metric == "set_f1":
        exp_order = _ordered_dates(expected)
        got_order = _ordered_dates(produced)
        exp, got = set(exp_order), set(got_order)
        missed = [d for d in exp_order if d not in got][:MAX_SPANS_PER_SIDE]
        wrong = [d for d in got_order if d not in exp][:MAX_SPANS_PER_SIDE]
        return _day_spans("missed", missed) + _day_spans("wrong", wrong)
    return []


def table_for(span: Span) -> str:
    """The feature table whose letters fit this span's length (F3)."""
    length = span.end - span.start
    if length <= DAILY_MAX:
        return "daily_feature"
    if length <= MONTHLY_MAX:
        return "monthly_feature"
    return "yearly_feature"


# -------------------------------------------------------------------- brackets
def bracket_sax(sax: str, window_start: datetime, window_end: datetime,
                span: Span, sax_len: int | None = None,
                width: int | None = None) -> str:
    """`sax` with the tokens whose time interval overlaps `span` in brackets.

    Token i covers `window_start + i * (window_end - window_start) / sax_len`
    up to the next token; a zero-length span brackets the single token that
    contains it. Nothing overlapping (the span lies outside the window) leaves
    the string untouched.

    E5: a token is `width` characters wide (1 for SAX, 2 for 1d-SAX / TFSAX,
    4 for MOSO) and occupies the characters [width*i, width*(i+1)), so the
    brackets can never cut a MOSO token in half. `width=None` reads the width
    off the string and `sax_len`, which gives exactly the pre-E5 behaviour for
    one-character symbols.
    """
    text = str(sax or "")
    w = int(width) if width else token_width(text, sax_len)
    w = max(1, w)
    n = int(sax_len or 0) or (len(text) // w)
    total = (window_end - window_start).total_seconds()
    if not text or n <= 0 or total <= 0:
        return text
    step = total / n
    first, last = None, None
    for i in range(min(n, len(text) // w)):
        lo = window_start + timedelta(seconds=i * step)
        # The last letter ends exactly at `window_end`: `i * step` is a float
        # and rounds to whole microseconds, which would otherwise leave a sliver
        # at the window border that belongs to no letter at all.
        hi = (window_end if i == n - 1
              else window_start + timedelta(seconds=(i + 1) * step))
        if span.is_instant:
            hit = lo <= span.start < hi
        else:
            hit = lo < span.end and hi > span.start
        if hit:
            first = i if first is None else first
            last = i
    if first is None or last is None:
        return text
    lo_c, hi_c = w * first, w * (last + 1)
    return f"{text[:lo_c]}[{text[lo_c:hi_c]}]{text[hi_c:]}"


# ------------------------------------------------------------------- rendering
def _fmt_duration(seconds: float) -> str:
    for unit, size in (("days", 86400.0), ("hours", 3600.0),
                       ("minutes", 60.0)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} seconds"


def letter_unit(window_start: datetime, window_end: datetime,
                n: int, width: int = 1) -> str:
    """What one letter / token of this row's symbolic string actually covers.

    Read off the row itself rather than off the table name, because a row may
    carry fewer tokens than the window has hours or days. E5: when a token is
    wider than one character the note says `token` and states the width, so
    that the reflector does not read a MOSO string as 4 letters per segment.
    """
    total = (window_end - window_start).total_seconds()
    if n <= 0 or total <= 0:
        return ""
    step = total / n
    word = _unit_word(width)
    for name, secs, tol in _UNIT_STEPS:
        if abs(step - secs) <= tol * secs:
            return f"1 {word} = {name}"
    return f"1 {word} = {_fmt_duration(step)}"


def _fmt_ts(value: datetime) -> str:
    return value.strftime(_TS_FMT)


def _fmt_window_start(raw: str, parsed: datetime) -> str:
    """A window start as written, without a pointless `00:00:00`."""
    text = str(raw).strip()
    if text.endswith(_MIDNIGHT):
        return text[:-len(_MIDNIGHT)]
    return text or _fmt_ts(parsed)


def _num(value: Any, *, signed: bool = False) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _symbol_cell(row: Dict[str, Any], column: str) -> Tuple[str, Any]:
    """The symbolic string of a row and its token count, whatever it is called.

    The reader aliases the configured column to its own name, but a row handed
    in by hand (or read from an older record) may still carry `sax` / `sym`, so
    all three spellings are accepted before giving up.
    """
    for key in (column, "sym", "sax"):
        if key in row and row.get(key) is not None:
            return str(row.get(key) or ""), row.get(f"{key}_len")
    return "", None


def render_row(table: str, row: Dict[str, Any], span: Span,
               column: str | None = None) -> str:
    """One feature-table row, with the span's tokens in brackets."""
    col = resolve_symbol_column(column)
    start = _parse_db_ts(row.get("window_start"))
    end = _parse_db_ts(row.get("window_end"))
    sax, sym_len = _symbol_cell(row, col)
    raw = sax
    width = token_width(raw, sym_len)
    unit = LETTER_UNIT.get(table, "1 letter = 1 segment")
    if width > 1:
        unit = unit.replace("1 letter", f"1 {_unit_word(width)}")
    if start is not None and end is not None:
        sax = bracket_sax(sax, start, end, span, sym_len, width)
        try:
            n = int(sym_len or 0) or (len(raw) // width)
        except (TypeError, ValueError):
            n = len(raw) // width
        unit = letter_unit(start, end, n, width) or unit
    try:
        rng = float(row["max_val"]) - float(row["min_val"])
        range_text = f"{rng:.3f}"
    except (KeyError, TypeError, ValueError):
        range_text = "n/a"
    where = (_fmt_window_start(str(row.get("window_start") or ""), start)
             if start is not None else str(row.get("window_start") or "?"))
    return (f"  {table} {where}  slope={_num(row.get('slope'), signed=True)} "
            f"std={_num(row.get('std_val'))} range={range_text}  "
            f"{col}={sax}   ({unit})")


def _parse_db_ts(value: Any) -> datetime | None:
    ts = parse_timestamp(value)
    if ts is None:
        return None
    return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts


def _is_row_line(line: str) -> bool:
    """Row lines are indented, span headers are not."""
    return line.startswith("  ")


def cap_chars(max_chars: int, width: int) -> int:
    """The character cap for a symbolization whose token is `width` wide (H5).

    `MAX_CHARS` was written for one-character SAX letters; a wider token makes
    every row proportionally longer without adding any information, so the cap
    scales with the width and every symbolization keeps the same number of
    spans and rows.
    """
    try:
        w = max(1, int(width))
    except (TypeError, ValueError):
        w = 1
    return int(max_chars) * w


def cap(text: str, max_chars: int = MAX_CHARS) -> str:
    """Whole lines only, never more than `max_chars` characters.

    A span header whose rows did not fit announces evidence that is not there,
    so it is dropped again — unless dropping it would leave nothing at all, in
    which case the bare header is still the more honest of the two.
    """
    if len(text) <= max_chars:
        return text
    kept: List[str] = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    while kept and not _is_row_line(kept[-1]) and any(map(_is_row_line, kept)):
        kept.pop()
    return "\n".join(kept)


# --------------------------------------------------------------------- builder
def read_only_uri(path: Path) -> str:
    """The `file:…?mode=ro` URI of `path`, with the path escaped.

    In a URI `?`, `#` and `%` are syntax: pasting a raw path behind `file:`
    makes SQLite open a different file (or none) as soon as a directory on the
    way holds one of them — and because every failure here is swallowed, the
    evidence block would then be missing for a whole run without a word.
    """
    return "file:" + quote(path.as_posix(), safe="/") + "?mode=ro"


class _Reader:
    """Read-only access to one task database, one query per table."""

    def __init__(self, conn: sqlite3.Connection, column: str | None = None):
        self.conn = conn
        # E5: the symbolic column may be called `sym`; asking for `sax` there
        # raises, and because every failure here is swallowed the reflector
        # would silently lose its evidence for a whole run.
        self.column = resolve_symbol_column(column)
        self._cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def rows(self, table: str, channel_id: str) -> List[Dict[str, Any]]:
        key = (table, channel_id)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        col = self.column
        try:
            cur = self.conn.execute(
                f"SELECT window_start, window_end, min_val, max_val, avg_val, "
                f"std_val, slope, {col}_len, {col} FROM {table} "
                f"WHERE channel_id = ? ORDER BY window_start", (channel_id,))
            rows = [dict(zip([c[0] for c in cur.description], r))
                    for r in cur.fetchall()]
        except sqlite3.Error:
            rows = []
        self._cache[key] = rows
        return rows

    def overlapping(self, table: str, channel_id: str, span: Span,
                    limit: int = MAX_ROWS_PER_SPAN) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in self.rows(table, channel_id):
            start = _parse_db_ts(row.get("window_start"))
            end = _parse_db_ts(row.get("window_end"))
            if start is None or end is None:
                continue
            hit = (start <= span.start < end if span.is_instant
                   else start < span.end and end > span.start)
            if hit:
                out.append(row)
            if len(out) >= max(1, int(limit)):
                break
        return out


def build_evidence(*, db_path: Path | str, question: str, eval_metric: str,
                   expected: Any, produced: Any,
                   max_chars: int = MAX_CHARS,
                   column: str | None = None) -> str:
    """The INDEX EVIDENCE block of one failed case, or `""` (ADDENDUM F3)."""
    try:
        spans = spans_of(eval_metric, expected, produced)
        if not spans:
            return ""
        path = Path(db_path)
        if not path.is_file():
            return ""
        conn = sqlite3.connect(read_only_uri(path), uri=True)
        try:
            channels = pick_channels(question, channel_ids_of(conn))
            if not channels:
                return ""
            # G4: one channel renders exactly as under F3; with two, each
            # block says which series it shows and brings one row per span.
            per_span = (MAX_ROWS_PER_SPAN if len(channels) == 1
                        else MAX_ROWS_PER_SPAN_MULTI)
            reader = _Reader(conn, column)
            lines: List[str] = []
            # ADDENDUM H5: the cap is token-width aware. A MOSO row carries the
            # same number of tokens as a SAX row in four times the characters,
            # so a fixed character cap would show the MOSO arm fewer spans than
            # the SAX arm for the very same case — a per-arm difference H3 says
            # must not exist. The width is read off the rows actually rendered
            # (1 for SAX, 2 for 1d-SAX / TFSAX, 4 for MOSO).
            width = 1
            for span in spans:
                table = table_for(span)
                for channel in channels:
                    rows = reader.overlapping(table, channel, span, per_span)
                    if not rows:
                        continue
                    label = (f"  [channel {channel}]" if len(channels) > 1
                             else "")
                    lines.append(f"{span.label}  {_fmt_ts(span.start)} → "
                                 f"{_fmt_ts(span.end)}{label}")
                    for row in rows:
                        width = max(width,
                                    token_width(*_symbol_cell(row,
                                                              reader.column)))
                        lines.append(render_row(table, row, span,
                                                reader.column))
            return cap("\n".join(lines), cap_chars(max_chars, width))
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — evidence is never worth a failed reflection
        return ""
