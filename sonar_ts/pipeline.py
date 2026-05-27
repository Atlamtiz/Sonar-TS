"""End-to-end orchestration for one task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from . import executor as executor_mod
from . import generator as gen
from . import planner as plan_mod
from .llm import DeepSeekClient
from .postprocess import format_prediction
from .schema import build_schema_text
from .skills import load_manifest, load_skills


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

    def __post_init__(self) -> None:
        if self.chosen_skills is None:
            self.chosen_skills = []
        if self.code_attempts is None:
            self.code_attempts = []
        if self.executor_log is None:
            self.executor_log = []


def run_task(task: Dict[str, Any], client: DeepSeekClient,
             *, executor_config: Dict[str, Any],
             skill_override: str | None = None) -> TaskRunRecord:
    rec = TaskRunRecord(task_id=task["id"])
    question: str = task["question"]
    eval_metric: str = task["eval_metric"]
    max_regen: int = executor_config.get("max_regenerations", 3)
    timeout: int = executor_config.get("timeout_seconds", 60)
    shape_rules: Dict[str, Dict] = executor_config.get("shape_rules", {})

    try:
        schema = build_schema_text(task["id"])
        if skill_override is not None:
            manifest = ("- id: coldstart\n"
                        "  description: methodology for this subtask")
        else:
            manifest = load_manifest()

        plan_result = plan_mod.plan(question, schema, manifest, client)
        rec.plan = plan_result.plan_text
        rec.chosen_skills = (["coldstart"] if skill_override is not None
                              else plan_result.skill_ids)
        experiences = (skill_override if skill_override is not None
                       else load_skills(plan_result.skill_ids))

        code = gen.generate(question, schema, rec.plan, experiences,
                            eval_metric, client)
        rec.code_attempts.append(code)

        result = executor_mod.execute(
            code, task["id"], eval_metric,
            timeout_seconds=timeout, shape_rules=shape_rules,
        )
        rec.executor_log.append({"status": result.status, "message": result.message})

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

    return rec
