"""Schema text builder."""

from __future__ import annotations

from .storage import ALL_TABLES, TaskDatabase


def build_schema_text(task_id: str) -> str:
    db = TaskDatabase(task_id)
    db.connect()
    try:
        lines = [f"# {task_id}"]
        for table in ALL_TABLES:
            cols = db.table_columns(table)
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
