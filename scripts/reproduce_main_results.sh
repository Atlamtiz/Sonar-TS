#!/usr/bin/env bash
# Reproduce the main results of Sonar-TS on NLQTSBench (the Sonar-TS row of the
# main results table): the MOSO index, the six-skill library, three runs averaged.
#
#   bash scripts/reproduce_main_results.sh            # everything
#   RUNS=1 bash scripts/reproduce_main_results.sh     # a single run
#   LIMIT=20 bash scripts/reproduce_main_results.sh   # smoke test on 20 tasks
#
# Needs configs/deepseek_api-key.txt (one key per line) and, for the full
# benchmark, the raw CSVs from HuggingFace (step 1 downloads them, ~1.7 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-3}"; LIMIT="${LIMIT:-0}"; WORKERS="${WORKERS:-}"
OUT="${OUT:-runs/main}"

# the configuration of the paper (also the defaults of the framework)
export SONAR_SYMBOLIZATION=moso SONAR_SYMBOL_COLUMN=sym SONAR_INDEX_GUIDE_DIR="$PWD/sonar_ts/prompts"

[ -f configs/deepseek_api-key.txt ] || { echo "configs/deepseek_api-key.txt is missing (copy the .example and paste your keys)"; exit 1; }

echo "== 1. raw series";        [ -d nlqtsbench/ts_data ] && [ "$(ls nlqtsbench/ts_data | wc -l)" -ge 1153 ] || python scripts/download_dataset.py
echo "== 2. per-task databases"; python -m scripts.load_benchmark
echo "== 3. feature tables (statistics, SAX and MOSO strings)"; python -m scripts.build_index

SUBSET=""
if [ "$LIMIT" != "0" ]; then
  python - "$LIMIT" <<'PY'
import json, sys
ids = [t["id"] for t in json.load(open("nlqtsbench/tasks.json"))][: int(sys.argv[1])]
json.dump(ids, open("runs_subset_ids.json", "w"))
PY
  SUBSET="--ids-file runs_subset_ids.json"
fi

for r in $(seq 1 "$RUNS"); do
  echo "== 4. run $r/$RUNS"
  python -m cold_start_stream.eval --tasks nlqtsbench/tasks.json --databases databases \
    --library PACKAGED --out "$OUT/r$r" ${WORKERS:+--workers "$WORKERS"} $SUBSET
done

echo "== 5. the main results row (mean of $RUNS run(s))"
python scripts/aggregate_runs.py $(for r in $(seq 1 "$RUNS"); do printf "%s/r%s " "$OUT" "$r"; done)
