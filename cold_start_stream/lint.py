"""Mechanical lint rules for skill drafts.

These run *before* the LLM linter and catch the violations that can be decided
without judgement: a literal year, an ISO date/timestamp, a training channel id
written as a literal, and — the `question_literal` rule of ADDENDUM A2 — a value
that one of the distilled questions *supplies* and that the skill therefore has
to read from the question instead of hard-coding. ADDENDUM F4 adds one more of
the same kind: a quoted run of at least 16 SAX letters is an index string
copied out of one window (`literal_sax_string`) — and, since E5, so is a run of
4+ MOSO tokens (`[a-e][RFS][012][=~]`) or 6+ two-character tokens
(`[a-e][A-E]`, i.e. 1d-SAX / TFSAX).

The split matters: the LLM linter now judges only beliefs about the data, so
every "did the model copy a number out of an example question?" decision is
made here, where it can be made exactly. It is also the only hard gate
(ADDENDUM B2), which is why ADDENDUM B1 tightened two of its rules: channel
ids are matched case-sensitively in two binding forms only, and a date that
carries a placeholder (`YYYY-01-01`, `<year>-01-01`) is a template, not a
literal.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any, Dict, Iterable, List, Sequence, Set

# ADDENDUM D3: the mechanical size limits live here, and `StreamConfig` takes
# its defaults from them. A run always passes `cfg.doc_max_chars` explicitly,
# but an audit or report script that calls `mechanical_lint(text)` directly
# gets the limit the run used instead of the retired 9000 — otherwise it
# reports `too_long` failures that the run never saw.
DEFAULT_DOC_MAX_CHARS = 12000
DEFAULT_SNIPPET_MAX_LINES = 100

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# An angle-bracket placeholder (`<year>`, `<start>`) may stand in for any part
# of a date, so the pattern matches template spellings as well as concrete
# dates; the exemption below then decides which of the two a hit is.
_PLACEHOLDER = r"<[A-Za-z_][^<>\s]{0,31}>"
_ISO_DATE_RE = re.compile(
    rf"(?<![\w-])(?:\d{{4}}|YYYY|{_PLACEHOLDER})"
    rf"-(?:\d{{2}}|MM|{_PLACEHOLDER})"
    rf"-(?:\d{{2}}|DD|{_PLACEHOLDER})"
    rf"(?:[ T](?:\d{{2}}|HH|{_PLACEHOLDER}):(?:\d{{2}}|MM|{_PLACEHOLDER})"
    rf"(?::(?:\d{{2}}|SS|{_PLACEHOLDER}))?)?")

# ADDENDUM B1: a date-like string that carries a placeholder letter run
# (`YYYY`, `MM`, `DD`, `HH`, `SS`) or an angle-bracket placeholder is a
# TEMPLATE, not a literal — `"YYYY-01-01"` and `"<year>-01-01"` say "fill this
# in from the question", which is exactly what the skill is asked to do. This
# subsumes the fixed format templates `YYYY-MM-DD[ HH:MM:SS]`.
_DATE_PLACEHOLDER_RE = re.compile(rf"YYYY|MM|DD|HH|SS|{_PLACEHOLDER}")


def _is_date_template(hit: str) -> bool:
    """Is this date-like hit a fill-in template rather than a literal date?"""
    return _DATE_PLACEHOLDER_RE.search(hit) is not None


def _context(text: str, start: int, end: int, width: int = 40) -> str:
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return " ".join(text[lo:hi].split())[:160]


def _channel_patterns(channel: str) -> List[re.Pattern[str]]:
    """The two forms in which a channel id counts as hard-coded (ADDENDUM B1).

    The id is matched CASE-SENSITIVELY and only as a whole token, either
    (a) inside quotes or backticks, or (b) directly after the word
    `channel`/`channel_id`. Anything looser turns ordinary prose into a
    violation: with a case-insensitive prose pattern the channel `OT` matched
    the "not" of "…not for single-channel extreme events…" in smoke_v2.
    The word `channel` itself may be capitalised; the id may not.
    """
    c = re.escape(channel)
    return [
        # (a) "OT" / 'OT' / `OT`
        re.compile(rf"""["'`]{c}["'`]"""),
        # (b) channel OT / channels: OT / channel_id = "OT"
        re.compile(rf"""(?i:channel(?:_id)?s?)\b[ \t]*(?:[=:]|==|!=|<>)?[ \t]*"""
                   rf"""["'`]?{c}(?![\w])"""),
    ]


# --------------------------------------------------------------- A2: question literals
# What a question SUPPLIES: the skill must read these from the question text,
# never carry them as a literal.
_Q_UPDOWN_RE = re.compile(r"\b([A-Za-z0-9]+_(?:up|down)_\d{4})\b", re.IGNORECASE)
_Q_CHANNEL_RE = re.compile(r"\bchannels?\b[^\w]{0,3}([A-Za-z0-9_]+)", re.IGNORECASE)
# The token after "channel" is often an ordinary English word ("the channel
# data", "the channel whose mean …") rather than an identifier. Reject those;
# keep anything that looks like an id — a digit, an upper-case code (`LULL`,
# `OT`) or an underscore name.
_Q_CHANNEL_STOPWORDS = frozenset({
    "the", "a", "an", "id", "ids", "under", "with", "named", "whose", "called",
    "data", "column", "columns", "values", "value", "series", "is", "are",
    "that", "it", "in", "of", "and", "or", "for", "from", "to", "each", "its",
    "this", "these", "those", "all", "any", "every", "name", "names", "pair",
    "pairs", "readings", "signal", "signals", "s",
})
_Q_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2})?(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
_Q_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_Q_DECIMAL_RE = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])(?!\.\d)")
_Q_NDAY_RE = re.compile(r"\b(\d{1,4})[-\s]days?\b", re.IGNORECASE)
# "the top-6 days where …" states a result count, not a 6-day window, even
# though it ends in "6 days"; without this the Composite-Trend phrasing of the
# benchmark turns every `pd.Timedelta(days=6)` into a violation.
_Q_COUNT_LEAD_RE = re.compile(
    r"\b(?:top|the|first|last|highest|lowest|largest|smallest|best|worst)"
    r"[-\s]$", re.IGNORECASE)
_Q_TOPN_RE = re.compile(r"\btop[-\s](\d{1,4})\b", re.IGNORECASE)
_Q_COUNT_RE = re.compile(
    r"\b(?:the|first|last|top|highest|lowest|largest|smallest)\s+(\d{1,4})\s+"
    r"(?:days?|dates?|timestamps?|intervals?|windows?|periods?|segments?|"
    r"points?|results?|channels?|rows?|values?|entries|items)\b", re.IGNORECASE)
_HAS_DIGIT_RE = re.compile(r"\d")

# Where a small integer is *binding* — i.e. it decides how many results come
# back or how long a window is, which is exactly what the question states.
#
# The two buckets are kept apart on purpose: a result count and a window
# length are different kinds of value, and a context that can only express one
# of them must not be checked against the other. Pairing them made the
# framework's own `str(ts)[:19]` / `str(ts)[:10]` date idiom a violation as
# soon as any pooled question happened to say "19-day", and made an ordinary
# `pd.Timedelta(days=2)` a violation as soon as one said "top-2".
_COUNT_TEMPLATES = (
    r"\btop[-\s]{v}\b",
    r"\bLIMIT\s+{v}\b",
    r"\bhead\(\s*{v}\s*\)",
    r"\bn(?:largest|smallest)\(\s*{v}\b",
    r"\[\s*\d*\s*:\s*{v}\s*\]",
)
_LENGTH_TEMPLATES = (
    r"\bn_days\s*=\s*{v}\b",
    r"\b{v}[-\s]days?\b",
    r"\bdays\s*=\s*{v}\b",
)
# A bare assignment states either kind of value, so it is checked against both.
_ASSIGN_TEMPLATES = (
    r"\b(?:top[_-]?k|top[_-]?n|n_top|k|n|limit)\s*=\s*{v}\b",
)

# A slice of a *string* can never be a result count: `str(ts)[:19]` and
# `str(ts)[:10]` are the timestamp/date idiom the runtime facts prescribe.
_STR_CALL_RE = re.compile(r"(?:\bstr|\.\s*strftime|\.\s*isoformat)\s*\($")
_STR_NAME_RE = re.compile(
    r"(?:\b\w*(?:str|txt|text|stamp|iso|label)\w*|[\"'])\s*$", re.IGNORECASE)


def _is_channel_token(token: str) -> bool:
    """Does the token after the word "channel" look like a channel id?"""
    if not token or token.lower() in _Q_CHANNEL_STOPWORDS:
        return False
    if _HAS_DIGIT_RE.search(token):
        return True
    # `LULL`, `OT`, `daily_feature`-style names; never a single letter, which
    # would match ordinary prose.
    return len(token) >= 2 and (token.isupper() or "_" in token)


def question_literals(questions: Iterable[str]) -> Dict[str, Set[str]]:
    """Split the values the questions supply into four buckets.

    * `anywhere` — dates, years, fractional thresholds and `<id>_up_<year>`
      series names: a skill may not contain them at all.
    * `channel`  — the channel id a question names. Short codes such as `OT`
      collide with ordinary English, so (ADDENDUM B1) they are looked for
      case-sensitively and only in the two hard-coding forms of
      `_channel_patterns`, not anywhere in the text.
    * `count`    — how many results the question asks for (`top-N`, "the N
      days"): ordinary small integers, so a violation only where they bind the
      size of the answer.
    * `length`   — how long a window the question states (`N-day`): likewise
      binding, but in length contexts only.
    """
    anywhere: Set[str] = set()
    channel: Set[str] = set()
    count: Set[str] = set()
    length: Set[str] = set()
    for question in questions:
        text = str(question or "")
        # `OT_up_2017` and friends cannot be an English word, so they stay in
        # the strict bucket.
        for m in _Q_UPDOWN_RE.finditer(text):
            anywhere.add(m.group(1))
        for m in _Q_CHANNEL_RE.finditer(text):
            if _is_channel_token(m.group(1)):
                channel.add(m.group(1))
        for rx in (_Q_DATETIME_RE, _Q_YEAR_RE, _Q_DECIMAL_RE):
            for m in rx.finditer(text):
                anywhere.add(m.group(0))
        for m in _Q_NDAY_RE.finditer(text):
            if _Q_COUNT_LEAD_RE.search(text[:m.start()]):
                count.add(m.group(1))  # "top-6 days" — a count, not a length
            else:
                length.add(m.group(1))
        for rx in (_Q_TOPN_RE, _Q_COUNT_RE):
            for m in rx.finditer(text):
                count.add(m.group(1))
    # A value that may not appear at all is not merely "binding".
    count -= anywhere | channel
    length -= anywhere | channel
    return {"anywhere": anywhere, "channel": channel,
            "count": count, "length": length}


def _anywhere_pattern(value: str) -> re.Pattern[str]:
    # `(?<![\w.])` keeps 67 out of 0.67 and of 1067; the trailing guards keep it
    # out of 67.5 while still matching "…4.287." at the end of a sentence.
    return re.compile(rf"(?<![\w.]){re.escape(value)}(?![\w])(?!\.\d)")


def _binding_patterns(value: str, templates: Iterable[str]
                      ) -> List[re.Pattern[str]]:
    v = re.escape(value)
    return [re.compile(t.format(v=v), re.IGNORECASE) for t in templates]


def _is_string_slice(text: str, start: int) -> bool:
    """Is the `[…:N]` at `start` slicing a string rather than a result list?

    `str(ts)[:19]` / `str(ts)[:10]` is the timestamp/date idiom the runtime
    facts mandate, so it can never be a hard-coded result count.
    """
    head = text[:start].rstrip()
    if head.endswith(")"):
        depth = 0
        for i in range(len(head) - 1, -1, -1):
            ch = head[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth -= 1
                if depth == 0:
                    return _STR_CALL_RE.search(head[:i + 1]) is not None
        return False
    return _STR_NAME_RE.search(head) is not None


def _question_literal_violations(skill_text: str,
                                 questions: Iterable[str]) -> List[Dict[str, str]]:
    literals = question_literals(questions)
    out: List[Dict[str, str]] = []
    for value in sorted(literals["anywhere"]):
        m = _anywhere_pattern(value).search(skill_text)
        if m is None:
            continue
        out.append({
            "quote": _context(skill_text, m.start(), m.end()),
            "why": (f"{value!r} is a value the questions state (a channel, a "
                    f"date, a year or a threshold); read it from the question "
                    f"instead of writing it into the skill"),
            "rule": "question_literal",
        })
    for value in sorted(literals["channel"]):
        for rx in _channel_patterns(value):
            m = rx.search(skill_text)
            if m is None:
                continue
            out.append({
                "quote": _context(skill_text, m.start(), m.end()),
                "why": (f"{value!r} is the channel a question names; read it "
                        f"from the question instead of writing it into the "
                        f"skill"),
                "rule": "question_literal",
            })
            break
    buckets = (
        ("count", _COUNT_TEMPLATES + _ASSIGN_TEMPLATES,
         "a result count the questions state"),
        ("length", _LENGTH_TEMPLATES + _ASSIGN_TEMPLATES,
         "a window length the questions state"),
    )
    reported: Set[str] = set()
    for bucket, templates, what in buckets:
        for value in sorted(literals[bucket], key=lambda s: (len(s), s)):
            if value in reported:
                continue
            for rx in _binding_patterns(value, templates):
                for m in rx.finditer(skill_text):
                    if "[" in m.group(0) and _is_string_slice(
                            skill_text, m.start()):
                        continue  # `str(ts)[:19]` — the mandated date idiom
                    out.append({
                        "quote": _context(skill_text, m.start(), m.end()),
                        "why": (f"{value!r} is {what}; take it from the "
                                f"question instead of binding the answer "
                                f"to it"),
                        "rule": "question_literal",
                    })
                    reported.add(value)
                    break
                if value in reported:
                    break
    return out


# --------------------------------------------------------------- C2: document sanity
# ADDENDUM C2. Seven of the 24 skills of full_v1 shipped with a code block the
# drafter never closed (it hit max_tokens); the generator then read half a
# script as if it were the method. These rules decide, without judgement,
# whether a document is a whole document.
_FENCE_LINE_RE = re.compile(r"(?m)^[ \t]*```")
_PY_BLOCK_RE = re.compile(
    r"(?ms)^([ \t]*)```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)^[ \t]*```",
    re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"(?ms)^([ \t]*)```[^\n]*\r?\n(.*?)^[ \t]*```")
_RESULT_ASSIGN_RE = re.compile(r"(?m)^[ \t]*_result\b[^=\n]*=(?!=)")
# A placeholder may be a whole phrase (`<year int>`, `<ordered kind list …>`):
# the released skills write them that way. It must start with a letter right
# after the `<` and stay on one line, so that `a < b and c > d` is arithmetic,
# not a placeholder.
_PLACEHOLDER_TOKEN_RE = re.compile(r"<[A-Za-z_][^<>\n]{0,120}>")
_QUOTES = ("'''", '"""', "'", '"')


def _is_document(text: str) -> bool:
    """Is this a whole skill document (frontmatter) or a mere fragment?"""
    return text.lstrip("﻿").lstrip().startswith("---")


def dedent_block(indent: str, body: str) -> str:
    """A fenced block as if its fence stood at column 0 (ADDENDUM G3).

    A fence that sits inside a numbered list is indented, and its body with
    it. `ast.parse` then reports "unexpected indent on line 1" and the whole
    document is rejected for a purely typographic reason — shape_v5/s1 lost a
    plateau skill that way, after two repair attempts that could not know what
    to repair. The fence's own indentation is stripped from every line that
    carries it, and `textwrap.dedent` takes whatever common indentation is
    left (a fence that is flush left but whose body is uniformly indented).
    """
    if indent:
        body = "".join(line[len(indent):] if line.startswith(indent) else line
                       for line in body.splitlines(keepends=True))
    return textwrap.dedent(body)


def python_blocks(skill_text: str) -> List[str]:
    """The bodies of the ```python fences of a skill document, dedented (G3)."""
    return [dedent_block(m.group(1), m.group(2))
            for m in _PY_BLOCK_RE.finditer(skill_text)]


def _fill_placeholders(code: str) -> str:
    """Replace `<placeholder>` with an identifier — outside strings only.

    Inside a string literal a placeholder is text the generated code fills in
    later (`'SELECT "<channel>" FROM raw_data'`), and replacing it there could
    turn valid Python into something else. A tiny scanner tracks the string
    state; it only has to be right about quotes and escapes.
    """
    out: List[str] = []
    i, n = 0, len(code)
    quote: str | None = None
    while i < n:
        ch = code[i]
        if quote is not None:
            if ch == "\\" and quote in ("'", '"'):
                out.append(code[i:i + 2])
                i += 2
                continue
            if code.startswith(quote, i):
                out.append(quote)
                i += len(quote)
                quote = None
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "#":                      # a comment runs to the end of line
            end = code.find("\n", i)
            end = n if end == -1 else end
            out.append(code[i:end])
            i = end
            continue
        for q in _QUOTES:
            if code.startswith(q, i):
                quote = q
                out.append(q)
                i += len(q)
                break
        else:
            m = _PLACEHOLDER_TOKEN_RE.match(code, i)
            if m is not None:
                out.append("_ph_" + _KEBAB_TO_NAME_RE.sub("_", m.group(0)[1:-1]))
                i = m.end()
                continue
            out.append(ch)
            i += 1
    return "".join(out)


_KEBAB_TO_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


_ID_SHAPED_RE = re.compile(r"[-_]")


# ------------------------------------------------------- F4: copied sax strings
# ADDENDUM F4. A run of the SAX letters a–e that is at least this long, inside
# a quoted string, is a string copied out of one example window: it can only
# ever match that one case, and it teaches the generator to compare strings
# instead of describing a shape. The run does not have to be the whole string:
# `LIKE '%aaaaaabbbbbbccccccdddddd%'` and `GLOB '*…*'` wrap the very same
# copied window in wildcards. Short illustrative strings ("aaaeee") stay
# allowed — the codebook is built on them — and a regular expression is not a
# plain letter run, so `[ab].*[de]` and `a{n,}` never match.
SAX_STRING_MIN_LEN = 16
# A single-line quoted string, per quote character (no escape handling: the
# strings we care about are plain letters and wildcards).
_QUOTED_SPAN_RE = re.compile(r"""\"([^\"\n]*)\"|'([^'\n]*)'|`([^`\n]*)`""")
# A maximal run of a–e letters inside such a string.
_SAX_RUN_RE = re.compile(rf"(?<![a-e])[a-e]{{{SAX_STRING_MIN_LEN},}}(?![a-e])")
SAX_REPAIR_MESSAGE = ("describe the pattern by letter classes and run lengths "
                      "instead of copying a string")

# E5. The same copied-window mistake in a symbolization whose "letter" is a
# multi-character TOKEN: a 16-letter run cannot occur there, because the token
# characters interrupt every letter run.
# ADDENDUM H5: the threshold is counted in SEGMENTS, and it is the same 16 for
# every symbolization. The earlier counts (4 MOSO tokens, 6 pair tokens) were
# chosen by character length, which made the gate 4x stricter per segment under
# MOSO than under SAX: a four-segment illustration written without the
# codebook's spaces lost its draft to a HARD gate where the SAX equivalent
# passed — a per-arm difference of the distillation pipeline that H3 says must
# not exist. Sixteen segments is still far more than any codebook example
# (at most five or six tokens, ALWAYS separated by spaces), while a string
# copied out of a window is one unbroken run of the whole window.
MOSO_TOKEN_MIN = 16         # [a-e][RFS][012][=~]  → 64 characters
PAIR_TOKEN_MIN = 16         # [a-e][A-E]           → 32 characters
_MOSO_RUN_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:[a-e][RFS][012][=~]){{{MOSO_TOKEN_MIN},}}")
# REVIEW of ADDENDUM H (2026-09-22): no trailing `(?![A-Za-z0-9])` here. The
# MOSO rule above never had one, and on the pair rule it did not tighten the
# gate, it PUNCHED A HOLE IN IT: the run cannot shrink below PAIR_TOKEN_MIN
# tokens, so one extra letter behind a copied string ('aAbBcCdDeEaAb', i.e. six
# tokens plus a stray level letter, as a `LIKE '%…%'` slice that ends inside a
# token easily is) made the whole match fail, and the copied string passed the
# HARD gate that the 12- and 14-character versions of the same string fail.
# The lookbehind alone is what keeps the run from starting inside a word.
_PAIR_RUN_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:[a-e][A-E]){{{PAIR_TOKEN_MIN},}}")
# (regex, characters per token, what to call one of them)
_TOKEN_RUN_RULES = ((_MOSO_RUN_RE, 4), (_PAIR_RUN_RE, 2))


def _sax_violation(skill_text: str, base: int, m: "re.Match[str]",
                   width: int) -> Dict[str, str]:
    value = m.group(0)
    what = (f"{len(value)}-letter" if width == 1
            else f"{len(value) // width}-token ({len(value)}-character)")
    return {
        "quote": _context(skill_text, base + m.start(), base + m.end()),
        "why": (f"the {what} string {value!r} is an index string copied from "
                f"one window; {SAX_REPAIR_MESSAGE}"),
        "rule": "literal_sax_string",
    }


def _literal_sax_violations(skill_text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for q in _QUOTED_SPAN_RE.finditer(skill_text):
        body = next((g for g in q.groups() if g is not None), "")
        if not body:
            continue
        base = q.start() + 1
        for rx, width in ((_SAX_RUN_RE, 1),) + _TOKEN_RUN_RULES:
            for m in rx.finditer(body):
                value = m.group(0)
                if value in seen:
                    continue
                seen.add(value)
                out.append(_sax_violation(skill_text, base, m, width))
    return out


def _other_skill_ref_pattern(sid: str) -> re.Pattern[str]:
    """Where naming `sid` counts as a reference to another skill.

    Ids are minted with `kebab()` over whatever the curator returns, so a
    library can hold a one-word id such as `anomaly` or `trend`. Matching
    those as a bare word turns ordinary prose ("an anomaly in the residual")
    into a violation of a HARD gate, for the rest of the run. A multi-word,
    hyphenated id cannot occur by accident and is matched anywhere; a
    single-word id only where it is quoted or introduced as a skill.
    """
    esc = re.escape(sid)
    if _ID_SHAPED_RE.search(sid):
        return re.compile(rf"(?<![A-Za-z0-9_-]){esc}(?![A-Za-z0-9_-])")
    return re.compile(
        rf"""(?:["'`]{esc}["'`]"""
        rf"""|(?i:skills?|use|see|call|via)[ \t]+["'`]?{esc})"""
        rf"""(?![A-Za-z0-9_-])""")


def _snippet_syntax_error(body: str) -> Exception | None:
    """The syntax error of a snippet, or None if it is valid Python.

    The snippet is parsed AS WRITTEN first and only then, if that failed, with
    `<placeholder>` spans replaced. `_fill_placeholders` cannot tell a
    placeholder from a `<` … `>` pair of comparisons on one line, so
    `if x < thr: print(y > 0)` would be swallowed into one identifier and
    valid Python would be reported as a syntax error (`snippet_syntax` is a
    hard gate). Parsing the raw text first makes the substitution a *rescue*
    for real placeholders instead of a mangling of correct code.
    """
    try:
        ast.parse(body)
        return None
    except SyntaxError as exc:
        first = exc
    except (ValueError, RecursionError) as exc:      # e.g. NUL bytes
        first = exc                                  # type: ignore[assignment]
    try:
        ast.parse(_fill_placeholders(body))
    except (SyntaxError, ValueError, RecursionError):
        return first
    return None


def _document_violations(skill_text: str, *, doc_max_chars: int,
                         snippet_max_lines: int,
                         other_skill_ids: Iterable[str] = ()
                         ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fences = _FENCE_LINE_RE.findall(skill_text)
    if len(fences) % 2 == 1:
        m = list(_FENCE_LINE_RE.finditer(skill_text))[-1]
        out.append({
            "quote": _context(skill_text, m.start(), m.end()),
            "why": (f"{len(fences)} code-fence lines: a ``` block is never "
                    f"closed, so the document was cut off"),
            "rule": "unclosed_fence",
        })

    blocks = python_blocks(skill_text)
    for body in blocks:
        exc = _snippet_syntax_error(body)
        if exc is None:
            continue
        if isinstance(exc, SyntaxError):
            out.append({
                "quote": " ".join(str(exc.text or body[:160]).split())[:160],
                "why": (f"the python snippet does not parse "
                        f"({exc.msg} on line {exc.lineno}); a skill's snippet "
                        f"must be valid Python once placeholders are filled"),
                "rule": "snippet_syntax",
            })
        else:                                        # e.g. NUL bytes
            out.append({"quote": " ".join(body[:160].split()),
                        "why": f"the python snippet cannot be parsed ({exc})",
                        "rule": "snippet_syntax"})

    # `no_result` is a statement about a WHOLE document: a prose fragment
    # (what most of the unit tests lint) is not missing anything.
    if (_is_document(skill_text)
            and not any(_RESULT_ASSIGN_RE.search(b) for b in blocks)):
        out.append({
            "quote": (" ".join(blocks[0][:160].split()) if blocks
                      else "(no python snippet)"),
            "why": ("no snippet assigns `_result`; the executor reads the "
                    "answer from that variable"),
            "rule": "no_result",
        })

    # ADDENDUM D3: the repair pass has to tell the drafter by how much it is
    # over, so the numbers travel with the violation instead of only sitting
    # in the prose of `why`.
    if len(skill_text) > doc_max_chars:
        out.append({
            "quote": f"document length {len(skill_text)} characters",
            "why": (f"the document is longer than {doc_max_chars} characters; "
                    f"a skill that long is pasted into every prompt and "
                    f"usually means fixes were appended instead of folded in"),
            "rule": "too_long",
            "over": "document",
            "size": len(skill_text),
            "limit": doc_max_chars,
        })
    for m in _ANY_BLOCK_RE.finditer(skill_text):
        body = dedent_block(m.group(1), m.group(2))
        n_lines = len(body.rstrip("\n").splitlines())
        if n_lines > snippet_max_lines:
            out.append({
                "quote": f"snippet of {n_lines} lines",
                "why": (f"a snippet longer than {snippet_max_lines} lines; "
                        f"keep one compact runnable example"),
                "rule": "too_long",
                "over": "snippet",
                "size": n_lines,
                "limit": snippet_max_lines,
            })
            break

    for sid in sorted({str(s) for s in other_skill_ids if str(s).strip()},
                      key=lambda s: (-len(s), s)):
        m = _other_skill_ref_pattern(sid).search(skill_text)
        if m is None:
            continue
        out.append({
            "quote": _context(skill_text, m.start(), m.end()),
            "why": (f"the document names another skill ({sid!r}); a skill is "
                    f"read on its own and must be self-contained"),
            "rule": "other_skill_ref",
        })
        break
    return out


def mechanical_lint(skill_text: str,
                    channel_literals: Iterable[str] = (),
                    questions: Iterable[str] = (),
                    other_skill_ids: Sequence[str] = (),
                    *, doc_max_chars: int = DEFAULT_DOC_MAX_CHARS,
                    snippet_max_lines: int = DEFAULT_SNIPPET_MAX_LINES
                    ) -> Dict[str, Any]:
    """Return `{"violations": [...], "verdict": "pass"|"fail"}`.

    `questions` are the questions of the group being distilled; whatever they
    supply may not survive as a literal in the skill (rule `question_literal`).
    `other_skill_ids` are the ids of the OTHER skills in the library — naming
    one of them breaks the "read on its own" contract (rule `other_skill_ref`).
    """
    violations: List[Dict[str, Any]] = []

    date_spans: List[tuple[int, int]] = []
    for m in _ISO_DATE_RE.finditer(skill_text):
        hit = m.group(0)
        date_spans.append((m.start(), m.end()))
        if _is_date_template(hit):
            continue
        violations.append({
            "quote": _context(skill_text, m.start(), m.end()),
            "why": (f"literal date/timestamp {hit!r}; use the fixed format "
                    f"template or a placeholder instead"),
            "rule": "literal_datetime",
        })

    for m in _YEAR_RE.finditer(skill_text):
        if any(lo <= m.start() < hi for lo, hi in date_spans):
            continue  # already reported as a date
        violations.append({
            "quote": _context(skill_text, m.start(), m.end()),
            "why": (f"literal year {m.group(0)!r}; a period a question states "
                    f"must be a placeholder read from the question"),
            "rule": "literal_year",
        })

    for channel in sorted({str(c) for c in channel_literals if str(c).strip()}):
        hits = [rx.search(skill_text) for rx in _channel_patterns(channel)]
        for m in hits:
            if m is None:
                continue
            violations.append({
                "quote": _context(skill_text, m.start(), m.end()),
                "why": (f"training channel id {channel!r} written as a literal; "
                        f"use the <channel> placeholder"),
                "rule": "training_channel",
            })
            break

    violations.extend(_literal_sax_violations(skill_text))
    violations.extend(_question_literal_violations(skill_text, questions))
    violations.extend(_document_violations(
        skill_text, doc_max_chars=doc_max_chars,
        snippet_max_lines=snippet_max_lines, other_skill_ids=other_skill_ids))

    # De-duplicate identical quotes (the year and date rules overlap).
    seen: set[tuple[str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for v in violations:
        key = (v["rule"], v["quote"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(v)

    return {"violations": unique,
            "verdict": "fail" if unique else "pass"}
