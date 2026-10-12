#!/usr/bin/env python3
"""Write a tasks.json of NLQTSBench-Styles for one linguistic style.

NLQTSBench-Styles (nlqtsbench/styles/nlqtsbench_styles.json) carries, for every
task of NLQTSBench, the question rewritten in four styles under
`question_styles`: lexical, technical, keyword, conversational. The answer key is
the original one. This script writes a task file in which `question` is the
rewrite of the chosen style, so that the benchmark runs unchanged on it:

    python scripts/make_style_tasks.py --style lexical
    python main.py --tasks nlqtsbench/styles/tasks_lexical.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STYLES = ("lexical", "technical", "keyword", "conversational")
HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--style", choices=STYLES, required=True)
    p.add_argument("--styles-file", default=str(HERE / "nlqtsbench" / "styles" / "nlqtsbench_styles.json"))
    p.add_argument("--out", default=None, help="Default: nlqtsbench/styles/tasks_<style>.json")
    a = p.parse_args()
    tasks = json.load(open(a.styles_file, encoding="utf-8"))
    out = []
    for t in tasks:
        q = t["question_styles"][a.style]
        row = {k: v for k, v in t.items() if k != "question_styles"}
        row["question"], row["original_question"], row["style"] = q, t["question"], a.style
        out.append(row)
    dest = Path(a.out) if a.out else HERE / "nlqtsbench" / "styles" / f"tasks_{a.style}.json"
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} tasks in the {a.style} style to {dest}")


if __name__ == "__main__":
    main()
