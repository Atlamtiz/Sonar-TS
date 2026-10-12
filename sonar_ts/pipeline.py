"""End-to-end orchestration for one task."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from . import executor as executor_mod
from . import generator as gen
from . import planner as plan_mod
from . import usage as usage_mod
from .llm import DeepSeekClient
from .postprocess import format_prediction
from .schema import build_schema_text
from .skills import SkillLibrary, load_manifest, load_skills


@dataclass
class TaskRunRecord:
    task_id: str
    plan: str = ""
    chosen_skills: list[str] = None  # type: ignore[assignment]
    code_attempts: list[str] = None  # type: ignore[assignment]
    executor_log: list[Dict[str, Any]] = None  # type: ignore[assignment]
    prediction: str = ""
    raw_result: Any = None
    final_status: str = "ok"
    final_message: str = ""
    planner_skills: list[str] = None  # type: ignore[assignment]
    # wall_s, llm_calls, prompt_tokens, completion_tokens, llm_time_s,
    # exec_time_s, exec_runs of this run (filled by `run_task`).
    timing: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.chosen_skills is None:
            self.chosen_skills = []
        if self.code_attempts is None:
            self.code_attempts = []
        if self.executor_log is None:
            self.executor_log = []
        if self.planner_skills is None:
            self.planner_skills = []
        if self.timing is None:
            self.timing = {}


def run_task(task: Dict[str, Any], client: DeepSeekClient,
             *, executor_config: Dict[str, Any],
             skill_override: str | None = None,
             library: SkillLibrary | None = None,
             forced_skill_ids: list[str] | None = None) -> TaskRunRecord:
    """Plan → generate → execute one task.

    `library` replaces the packaged skill library for both the planner's
    manifest and the generator's skill bodies. `forced_skill_ids` still lets
    the planner write the plan but hands the generator exactly those skills
    (the planner's own pick is kept in `planner_skills`). `skill_override`
    keeps its old meaning and wins over both.
    """
    rec = TaskRunRecord(task_id=task["id"])
    question: str = task["question"]
    eval_metric: str = task["eval_metric"]
    max_regen: int = executor_config.get("max_regenerations", 3)
    timeout: int = executor_config.get("timeout_seconds", 60)
    shape_rules: Dict[str, Dict] = executor_config.get("shape_rules", {})

    wall_started = time.perf_counter()
    usage_token = usage_mod.begin_task(rec.task_id)
    exec_elapsed: list[float] = []
    exec_sql: list[float] = []
    try:
        schema = build_schema_text(task["id"])
        if skill_override is not None:
            manifest = ("- id: coldstart\n"
                        "  description: methodology for this subtask")
        elif library is not None:
            manifest = library.manifest()
        else:
            manifest = load_manifest()

        plan_result = plan_mod.plan(question, schema, manifest, client)
        rec.plan = plan_result.plan_text
        rec.planner_skills = list(plan_result.skill_ids)

        if skill_override is not None:
            rec.chosen_skills = ["coldstart"]
            experiences = skill_override
        else:
            if forced_skill_ids is not None:
                rec.chosen_skills = list(forced_skill_ids)
            else:
                rec.chosen_skills = list(plan_result.skill_ids)
            experiences = (library.load(rec.chosen_skills) if library is not None
                           else load_skills(rec.chosen_skills))

        code = gen.generate(question, schema, rec.plan, experiences,
                            eval_metric, client)
        rec.code_attempts.append(code)

        result = executor_mod.execute(
            code, task["id"], eval_metric,
            timeout_seconds=timeout, shape_rules=shape_rules,
        )
        rec.executor_log.append({"status": result.status, "message": result.message})
        exec_elapsed.append(getattr(result, "elapsed_s", 0.0))
        exec_sql.append(getattr(result, "sql_s", 0.0))

        attempt = 0
        while not result.is_ok() and attempt < max_regen:
            attempt += 1
            code = gen.regenerate(
                question, schema, rec.plan, experiences, eval_metric,
                previous_code=code,
                error_status=result.status,
                error_message=result.message,
                client=client,
            )
            rec.code_attempts.append(code)
            result = executor_mod.execute(
                code, task["id"], eval_metric,
                timeout_seconds=timeout, shape_rules=shape_rules,
            )
            rec.executor_log.append({"status": result.status, "message": result.message})
            exec_elapsed.append(getattr(result, "elapsed_s", 0.0))
            exec_sql.append(getattr(result, "sql_s", 0.0))

        if result.is_ok():
            rec.raw_result = result.result
            rec.prediction = format_prediction(task, result.result)
            rec.final_status = "ok"
        else:
            rec.final_status = result.status
            rec.final_message = result.message
    except Exception as exc:  # noqa: BLE001
        rec.final_status = "pipeline_error"
        rec.final_message = f"{type(exc).__name__}: {exc}"
    finally:
        rec.timing = usage_mod.task_timing(
            usage_mod.current_usage(),
            wall_s=time.perf_counter() - wall_started,
            exec_elapsed=exec_elapsed, exec_sql=exec_sql)
        usage_mod.end_task(usage_token)

    return rec
