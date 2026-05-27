"""Per-subtask cold-start iteration loop."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from sonar_ts.llm import DeepSeekClient
from sonar_ts.pipeline import run_task
from sonar_ts.evaluator import score_one

from .agents import CriticAgent, DrafterAgent, ReflectorAgent
from .agents.reflector import TaskExample


@dataclass
class VerifyResult:
    score: float
    per_task: List[Dict[str, Any]] = field(default_factory=list)


def _verify(skill_body: str, tasks: List[Dict[str, Any]],
            llm_cfg: Dict[str, Any], exec_cfg: Dict[str, Any],
            api_keys: List[str], n_workers: int) -> VerifyResult:
    q: "queue.Queue" = queue.Queue()
    for t in tasks:
        q.put(t)
    n_workers = max(1, min(n_workers, len(tasks)))
    lock = threading.Lock()
    out: List[Dict[str, Any]] = []

    def _worker(thread_id: int, key: str) -> None:
        client = DeepSeekClient(
            api_key=key, base_url=llm_cfg["base_url"],
            model=llm_cfg["model"], temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
            thinking_mode=llm_cfg.get("thinking_mode", "disabled"),
            max_attempts=llm_cfg["http_max_attempts"],
            backoff_seconds=llm_cfg["http_backoff_seconds"],
        )
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = run_task(t, client, executor_config=exec_cfg,
                               skill_override=skill_body)
                prediction = rec.prediction
                status = rec.final_status
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[verify t{thread_id}] {t['id']} "
                                 f"{type(exc).__name__}: {exc}\n")
                prediction, status = "", "pipeline_crash"
            try:
                score, _br = score_one(t["eval_metric"], prediction,
                                        t.get("ground_truth", t.get("answer")))
            except Exception:  # noqa: BLE001
                score = 0.0
            with lock:
                out.append({
                    "task_id":     t["id"],
                    "question":    t["question"],
                    "ground_truth": t.get("ground_truth", t.get("answer")),
                    "prediction":  prediction,
                    "score":       float(score),
                    "status":      status,
                    "eval_metric": t["eval_metric"],
                })
            q.task_done()

    threads: List[threading.Thread] = []
    for i in range(n_workers):
        th = threading.Thread(target=_worker, name=f"verify-w{i}",
                              args=(i, api_keys[i % len(api_keys)]),
                              daemon=False)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    mean = sum(r["score"] for r in out) / len(out) if out else 0.0
    return VerifyResult(score=mean, per_task=out)


def _pick_examples(per_task: List[Dict[str, Any]],
                   n_fail: int = 6, n_pass: int = 2
                   ) -> tuple[List[TaskExample], List[TaskExample]]:
    by_score = sorted(per_task, key=lambda r: r["score"])
    fails  = [r for r in by_score if r["score"] < 0.5][:n_fail]
    passes = [r for r in by_score[::-1] if r["score"] >= 0.7][:n_pass]

    def to_ex(r: Dict[str, Any]) -> TaskExample:
        return TaskExample(
            question=r["question"],
            ground_truth=str(r["ground_truth"]),
            prediction=str(r["prediction"]),
            score=r["score"],
            eval_metric=r["eval_metric"],
        )
    return [to_ex(r) for r in fails], [to_ex(r) for r in passes]


@dataclass
class RoundReport:
    v: int
    score: float
    n_examples: int


def cold_start_one_subtask(
    subtask: str, eval_metric: str,
    train_tasks: List[Dict[str, Any]],
    llm_cfg: Dict[str, Any],
    exec_cfg: Dict[str, Any],
    api_keys: List[str],
    *,
    max_rounds: int = 3,
    plateau_delta: float = 0.01,
    workers: int = 10,
    traces_dir: Path,
    out_skill_path: Path,
) -> List[RoundReport]:
    if not train_tasks:
        print(f"  [{subtask}] no training tasks available — skipping.")
        return []

    reflector_client = DeepSeekClient(
        api_key=api_keys[0], base_url=llm_cfg["base_url"],
        model=llm_cfg["model"], temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        thinking_mode=llm_cfg.get("thinking_mode", "disabled"),
        max_attempts=llm_cfg["http_max_attempts"],
        backoff_seconds=llm_cfg["http_backoff_seconds"],
    )
    reflector = ReflectorAgent(reflector_client)
    drafter   = DrafterAgent(reflector_client)
    critic    = CriticAgent(reflector_client)

    traces_dir.mkdir(parents=True, exist_ok=True)
    skill_body = ""
    prev_score = -1.0
    best_score = -1.0
    best_skill = ""
    best_v = 0
    rounds: List[RoundReport] = []

    for v in range(1, max_rounds + 1):
        round_dir = traces_dir / f"v{v}"
        round_dir.mkdir(exist_ok=True)
        print(f"\n  [{subtask}] round v{v} — verifying current skill on "
              f"{len(train_tasks)} training tasks…")
        t0 = time.time()
        verify = _verify(skill_body, train_tasks, llm_cfg, exec_cfg,
                          api_keys, workers)
        elapsed = time.time() - t0
        print(f"  [{subtask}] v{v} verify score = {verify.score:.4f}  "
              f"({elapsed:.1f}s)")
        (round_dir / "verify_scores.json").write_text(
            json.dumps({"mean_score": verify.score,
                        "per_task":   verify.per_task}, indent=2),
            encoding="utf-8")
        rounds.append(RoundReport(v=v, score=verify.score,
                                  n_examples=len(verify.per_task)))

        if verify.score > best_score:
            best_score = verify.score
            best_skill = skill_body
            best_v = v

        if v > 1 and verify.score - prev_score < plateau_delta:
            print(f"  [{subtask}] plateau detected "
                  f"(Δ={verify.score - prev_score:+.4f} < {plateau_delta}); "
                  f"stopping at v{v}.")
            break
        prev_score = verify.score

        fails, passes = _pick_examples(verify.per_task)
        (round_dir / "reflector_input.json").write_text(
            json.dumps({"failures": [vars(f) for f in fails],
                        "successes": [vars(s) for s in passes]}, indent=2),
            encoding="utf-8")

        print(f"  [{subtask}] reflector: {len(fails)} fail + "
              f"{len(passes)} pass …")
        patterns = reflector.reflect(subtask, eval_metric, fails, passes)
        (round_dir / "reflector_output.md").write_text(patterns,
                                                       encoding="utf-8")

        print(f"  [{subtask}] drafter v1 …")
        draft = drafter.draft(subtask, eval_metric, patterns, skill_body)
        (round_dir / "drafter_v1.md").write_text(draft, encoding="utf-8")

        print(f"  [{subtask}] critic …")
        critique = critic.review(subtask, eval_metric, draft,
                                  fails[:3] + passes[:1])
        (round_dir / "critic.md").write_text(critique, encoding="utf-8")

        print(f"  [{subtask}] drafter v2 (revise) …")
        skill_body = drafter.revise(subtask, draft, critique)
        (round_dir / "drafter_v2.md").write_text(skill_body, encoding="utf-8")

    out_skill_path.parent.mkdir(parents=True, exist_ok=True)
    out_skill_path.write_text(best_skill, encoding="utf-8")
    print(f"  [{subtask}] best v{best_v} (score={best_score:.4f}) "
          f"→ {out_skill_path}")
    return rounds
