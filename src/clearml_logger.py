"""Optional ClearML experiment tracking — config-gated, kept out of
train.py so the main training script stays readable without ClearML noise.

ClearML auto-hooks the TensorBoard writer HF Trainer already uses
(`report_to=["tensorboard"]`) once a Task is initialized — no custom
Trainer callback needed, just start/close the Task around the run.

`clearml` is an optional dependency (`pip install -e .[clearml]`) — imported
lazily so nothing breaks when `clearml.enabled: false`.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def init_clearml(cfg: dict[str, Any], run_id: str) -> Any | None:
    """Start a ClearML Task if `clearml.enabled` is true in config. Returns
    the Task, or None when disabled (callers must handle both)."""
    clearml_cfg = cfg.get("clearml", {})
    if not clearml_cfg.get("enabled", False):
        return None

    from clearml import Task

    task = Task.init(
        project_name=clearml_cfg.get("project_name", "contradictory-my-dear-watson"),
        task_name=clearml_cfg.get("task_name") or run_id,
    )
    task.connect(cfg)
    logger.info("ClearML task started: %s", task.get_output_log_web_page())
    return task


def close_clearml(task: Any | None) -> None:
    if task is not None:
        task.close()
