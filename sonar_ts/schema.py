"""Schema text builder."""

from __future__ import annotations

from .storage import ALL_TABLES, TaskDatabase


def build_schema_text(task_id: str) -> str:
    # Read-only: this runs for every item of a run, against database trees the
    # project may declare read-only.
    db = TaskDatabase(task_id, read_only=True)
    db.connect()
    try:
        # Header stays neutral on purpose: the task id spells out the
        # benchmark sub-task, which must never reach a prompt.
        lines = ["# task database"]
        for table in ALL_TABLES:
            cols = db.table_columns(table)
            if not cols:          # the table does not exist in this database
                continue
            formatted_cols = []
            for c in cols:
                if table == "raw_data" and c != "timestamp":
                    formatted_cols.append(f'"{c}"')
                else:
                    formatted_cols.append(c)
            lines.append(f"# Table: {table} [{', '.join(formatted_cols)}]")
        lines.append("#")
        lines.append(
            "# Note: each non-timestamp column in raw_data is a channel; "
            "the column names appear as channel_id values in the feature tables."
        )
        return "\n".join(lines)
    finally:
        db.close()
