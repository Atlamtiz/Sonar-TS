# FRAMEWORK RUNTIME — facts every skill must respect

Generated code runs in a subprocess with these globals already bound:

    conn   — sqlite3.Connection to the task's database (read-only)
    pd, np, sqlite3, json, math, re, datetime, timedelta

Each task database has four tables.

- `raw_data` — WIDE. A "timestamp" column plus one column per channel; the column
  name IS the channel identifier. Double-quote it in SQL:
      SELECT timestamp, "<channel>" FROM raw_data
       WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp
  raw_data can hold more than one value column. Besides a plain channel column
  there may be a COMPANION column whose name is the channel id plus a suffix:
  `<channel>_hist_mean` holds that channel's historical mean curve for the same
  timestamps (its usual seasonal level). Two series that belong together appear as
  two columns of the same table (for example an upstream and a downstream series).
  Always look at the actual column list of the task database before deciding what
  to read.
- `yearly_feature`, `monthly_feature`, `daily_feature` — LONG index tables, one row
  per (channel, window):
      channel_id, window_start (inclusive), window_end (exclusive),
      min_val, max_val, avg_val, std_val, slope, <<SYMCOL>>_len, <<SYMCOL>>

# INDEX GUIDE — the planner and the code generator ALWAYS see this text, with or without a skill

<<INDEX_GUIDE>>

# OTHER RUNTIME RULES

TIME LIMIT. The script is killed after 60 seconds of wall time and then scores
zero. Write vectorised pandas / numpy (rolling, groupby, cumulative sums, array
slicing); never loop in Python over every row, every timestamp or every candidate
window of a long series when a vectorised form exists; shortlist with the feature
tables first and run expensive verification only on the shortlist; cap the number
of candidates you verify. A method that is exact but too slow is a wrong method.

The script must END by assigning `_result`. Its shape is fixed by the task's
eval_metric:

    rel_acc → a number
    hit     → one timestamp string "YYYY-MM-DD HH:MM:SS"
    iou     → [start_timestamp, end_timestamp]
    set_f1  → a list of date strings ["YYYY-MM-DD", ...] (at most 50)
    report  → {"trend_segments": [{"start","end","adj","kind"}, ...],
               "outliers": [{"timestamp","value"}, ...]}

Convert numpy / pandas values to native Python (.item(), .tolist(), str(ts))
before assigning. Never refer to inputs the framework does not provide
(there is no df_pred, df_true, or pre-loaded DataFrame).
