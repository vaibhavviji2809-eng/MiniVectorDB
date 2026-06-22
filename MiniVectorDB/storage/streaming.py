"""Streaming insert pipeline with background commits."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class InsertPipeline(Generic[T]):
    commit_callback: Callable[[T], None]
    batch_size: int = 64
    merge_interval: float = 0.25
    _queue: "queue.Queue[tuple[T, threading.Event | None]]" = field(
        default_factory=queue.Queue, init=False
    )
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _worker: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="MiniVectorDBInsertPipeline", daemon=True)
        self._worker.start()

    def submit(self, item: T, wait: bool = True) -> T:
        self.start()
        ack = threading.Event() if wait else None
        self._queue.put((item, ack))
        if ack is not None:
            ack.wait()
        return item

    def flush(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _run(self) -> None:
        batch: list[tuple[T, threading.Event | None]] = []
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=self.merge_interval)
                batch.append(item)
            except queue.Empty:
                pass

            if batch and (len(batch) >= self.batch_size or self._queue.empty()):
                for payload, ack in batch:
                    self.commit_callback(payload)
                    if ack is not None:
                        ack.set()
                    self._queue.task_done()
                batch.clear()
