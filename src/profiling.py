"""Config-gated torch.profiler hook for HF Trainer.

Off by default (zero overhead). Enable via config `profiling.enabled: true`
to capture a chrome trace of the first N training steps, used to find
bottlenecks (dataloader vs forward/backward vs optimizer) on the 5090.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class ProfilerCallback(TrainerCallback):
    def __init__(self, output_dir: Path, profile_steps: int) -> None:
        self.output_dir = output_dir
        self.profile_steps = profile_steps
        self._prof: torch.profiler.profile | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=self.profile_steps),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            with_stack=False,
        )
        self._prof.start()
        logger.info(
            "profiler started, capturing first %d steps to %s", self.profile_steps, self.output_dir
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._prof is not None:
            self._prof.step()
            if state.global_step >= self.profile_steps + 1:
                self._prof.stop()
                logger.info("profiler trace written to %s (view with tensorboard)", self.output_dir)
                self._prof = None

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._prof is not None:
            self._prof.stop()
            self._prof = None
