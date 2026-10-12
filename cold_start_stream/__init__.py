"""Streaming, label-free cold start (v1).

A shuffled, unlabeled stream of training questions is solved batch by batch
with the current skill library; failures are grouped by the skill that was
loaded, diagnosed, turned into a skill patch or a new skill, gated on the very
items that failed, and committed only if the method actually helps.

The old per-sub-task `cold_start/` package is untouched and stays as reference.
"""

from .config import StreamConfig
from .orchestrator import StreamOrchestrator
from .stream import StreamItem, load_stream

__all__ = ["StreamConfig", "StreamOrchestrator", "StreamItem", "load_stream"]
