"""Configuration of the streaming cold start."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from .lint import DEFAULT_DOC_MAX_CHARS, DEFAULT_SNIPPET_MAX_LINES


@dataclass
class StreamConfig:
    """Every knob of the streaming procedure (all overridable from the CLI)."""

    batch_size: int = 25
    fail_thr: float = 0.5
    pass_thr: float = 0.7
    # ADDENDUM C4: gate on enough items — three items decided nothing in v1.
    # ADDENDUM D4: the threshold now depends on what a trigger would do.
    # A NONE/SPLIT group writes a NEW skill, which is expensive and must be
    # backed by evidence; a skill with a real replay set can be patched on
    # fewer failures, because the replay set protects what already works.
    trigger_min_fail_create: int = 5
    trigger_min_fail_patch: int = 3
    # …but only once that skill has enough practice; below this the PATCH
    # threshold falls back to `trigger_min_fail_create`.
    patch_replay_min: int = 6
    final_min_fail: int = 3
    max_cases_reflect: int = 8          # C3
    max_success_cases: int = 2
    group_max_items: int = 60
    max_triggers_per_batch: int = 8     # C5
    gate_fail_cap: int = 12             # C4
    replay_cap: int = 12
    accept_delta: float = 0.10
    replay_tol: float = 0.05
    # G2: a candidate that passes lint but misses the gate gets this many
    # gate-feedback revision rounds before it is rejected (0 = the old
    # behaviour: rejected at once). One intent was drafted 9 times in three
    # streams of shape_v5 and thrown away 9 times, each time from scratch.
    # G5: the default is 1 — in shape_v5b 7 of the 14 acceptances came in
    # round 1 and NONE in round 2 (4 of 6 round-2 attempts died on
    # `too_long`), so a second round only inflates the documents.
    gate_revision_rounds: int = 1
    max_attempts_per_item: int = 2
    code_max_lines: int = 120
    seed: int = 20260920
    epochs: int = 1
    workers: int = 64                   # C5
    # C5: how many triggers of one batch are distilled at the same time
    # (1 = the old sequential behaviour); never more than one per trigger.
    trigger_workers: int = 8
    # C2: the distillation agents write whole documents and must not be cut off.
    distill_max_tokens: int = 8192
    # C3: what the distiller is shown per case.
    gt_max_chars: int = 4000
    max_reference_solutions: int = 2
    # C2: hard mechanical limits on a skill document.
    # D3: 9000 rejected 20 candidates whose documents were 9.5–11.7 k.
    doc_max_chars: int = DEFAULT_DOC_MAX_CHARS
    snippet_max_lines: int = DEFAULT_SNIPPET_MAX_LINES
    # D5: a group of items whose PATCH was rejected is a SPLIT candidate when
    # this share of it failed under the SAME skill.
    split_min_share: float = 0.6
    # D5: after a split CREATE, this share of its items must route to it.
    split_routing_min_hit: float = 0.6
    # C6: library budget and consolidation.
    library_budget: int = 16
    exam_cap: int = 30
    refine_delta: float = 0.02
    refine_rounds: int = 1
    routing_min_hit: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
