"""Drafter agent: write or revise a per-subtask skill markdown."""

from __future__ import annotations

from sonar_ts.llm import DeepSeekClient


_DRAFT_SYSTEM = """You are writing a `methodology skill` for a Sonar-TS
sub-task. The framework will inject this markdown verbatim into the
code-generator prompt, so the markdown's claims about the runtime
environment MUST match reality.

# FRAMEWORK RUNTIME — facts you MUST respect

The generated code runs in a subprocess with these globals already bound:

    conn       — sqlite3.Connection to the task's database (read-only)
    pd         — pandas
    np         — numpy
    sqlite3, json, math, re, datetime, timedelta

The database has four tables:

    raw_data           — WIDE: columns are (timestamp, <channel_1>, <channel_2>, ...)
                          Channel names may need double-quoting in SQL.
    sax_1h, sax_1d, sax_1w  — three multi-scale SAX feature tables (long format)

Read raw values with:
    df = pd.read_sql_query(
        'SELECT timestamp, "<channel>" FROM raw_data WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp',
        conn, params=(start, end))
    df["timestamp"] = pd.to_datetime(df["timestamp"])

The code MUST end by assigning to the variable `_result` (the framework
appends `print(json.dumps({"_result": _result}))` after it). NEVER mention
`df_pred`, `df_true`, or any input DataFrames — the framework does NOT
provide them.

# _result SHAPE per eval_metric — non-negotiable

| eval_metric | _result must be …                                       |
|-------------|---------------------------------------------------------|
| rel_acc     | a number (int or float)                                 |
| hit         | one ISO timestamp string "YYYY-MM-DD HH:MM:SS"          |
| iou         | a list of two ISO timestamp strings [start, end]        |
| set_f1      | a list of date strings ["YYYY-MM-DD", ...] (≤ 50)       |
| report      | a dict {"trend_segments": [...], "outliers": [...]}     |

# OUTPUT FORMAT — STRICT

    ---
    id: <kebab-case-id>
    description: <one-sentence summary of the methodology>
    ---

    # <Subtask name> — <short tagline>

    ## Algorithm

    <step-by-step methodology with a runnable Python snippet using
     `conn`, `pd`, `np`. The snippet should END with `_result = ...`
     in the exact shape required by the eval_metric above.>

    ## Rules

    <bulleted invariants the generator must respect>

# GUIDELINES

- Be SPECIFIC: name concrete pandas/numpy operations, window sizes,
  thresholds. Vague prose ("smooth appropriately") is useless.
- Quote channel names in SQL: `"<channel_name>"` — channels can start
  with digits or contain hyphens.
- If the previous skill has working content for some cases, KEEP what
  works and add/replace only what the error patterns indicate.
- No preamble, no commentary — emit the skill markdown directly.
"""


_REVISE_SYSTEM = """You are revising a `methodology skill` based on a
Critic's review. Apply each piece of feedback IN PLACE — keep what the
Critic likes, change what they flag.

Output the FULL revised skill in the same strict format the Drafter uses
(frontmatter + ## Output schema + ## Algorithm + ## Rules). No preamble.
"""


class DrafterAgent:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def draft(self, subtask: str, eval_metric: str, error_patterns: str,
              prev_skill: str) -> str:
        """First-pass skill drafting from observed error patterns."""
        prev = prev_skill.strip()
        prev_block = (f"### Previous skill (v{'0' if not prev else 'N'}):\n"
                      f"{prev or '(empty — true cold start)'}\n")
        user = (
            f"Sub-task: **{subtask}**\n"
            f"Eval metric: `{eval_metric}`\n\n"
            f"### Error patterns to address:\n{error_patterns}\n\n"
            f"{prev_block}\n"
            "Write the next skill version. Output the markdown only."
        )
        return self.client.chat([
            {"role": "system", "content": _DRAFT_SYSTEM},
            {"role": "user",   "content": user},
        ]).strip()

    def revise(self, subtask: str, draft_skill: str, critique: str) -> str:
        """Apply Critic feedback to a draft."""
        user = (
            f"Sub-task: **{subtask}**\n\n"
            f"### Current draft:\n{draft_skill}\n\n"
            f"### Critic feedback:\n{critique}\n\n"
            "Apply the feedback. Output the full revised skill only."
        )
        return self.client.chat([
            {"role": "system", "content": _REVISE_SYSTEM},
            {"role": "user",   "content": user},
        ]).strip()
