# Prompt cold-start

Bootstrap one methodology skill (`*.md`) per sub-task by running a multi-agent loop (Reflector → Drafter → Critic → Drafter-revise) on a training set that's disjoint from the test set.

## Quick start

```bash
# 1. Pull the training set (~1 min,)
python -m cold_start.download_train_data

# 2. Run the loop (~40 min, ~$1)
python -m cold_start.run_cold_start```
```

## Output



cold_start/
├── discovered_skills/<Subtask>.md    ★ the synthesised skill (best-scoring round)
├── traces/<Subtask>/v<N>/             per-round audit (verify scores, agent I/O)
└── summary.json                       score-vs-round per sub-task



> Discovered skills are **not** auto-installed — diff against `sonar_ts/skills/library/<id>.md` and copy over manually.

