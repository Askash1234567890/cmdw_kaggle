"""Async TensorBoard logging callback for HF Trainer.

The default `TensorBoardCallback` (auto-attached via `report_to=["tensorboard"]`)
writes scalars synchronously on the main thread during `on_log`, right after
the `tr_loss.item()` CUDA sync — stalling the main thread (and therefore GPU
kernel dispatch) for the duration of the disk write. This callback offloads
the actual `SummaryWriter.add_scalar` calls to a background thread via a
queue, so `on_log` only has to enqueue already-materialized python floats
and return immediately.

Always active, not config-gated — pure I/O optimization, doesn't change
training behavior or logged values.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

_SENTINEL = None


class AsyncTensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._writer: SummaryWriter | None = None
        self._queue: queue.Queue[tuple[str, float, int] | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def _writer_loop(self) -> None:
        assert self._writer is not None
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            tag, value, step = item
            self._writer.add_scalar(tag, value, step)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logger.info("async tensorboard writer started, log_dir=%s", self.log_dir)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        for tag, value in logs.items():
            if isinstance(value, (int, float)):
                self._queue.put((tag, value, state.global_step))

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._thread is None:
            return
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            logger.warning("async tensorboard writer thread did not finish within timeout, some logs may be lost")
        if self._writer is not None:
            self._writer.close()
        logger.info("async tensorboard writer stopped")
