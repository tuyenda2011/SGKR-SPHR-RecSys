"""Flushed terminal/file progress and a heartbeat for long synchronous stages."""
import math
import sys
import threading
import time
from pathlib import Path

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)


class TrainingProgress:
    def __init__(self, label, interval=10., path=None, stream=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Progress interval must be positive and finite")
        self.label, self.interval = label, interval
        self.stream = stream if stream is not None else sys.stdout
        self.path = Path(path) if path is not None else None
        self.started = time.monotonic()
        self.last_print = self.started
        self.message = "starting"
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = None

    def _emit(self, message):
        line = f"[{self.label} +{time.monotonic()-self.started:.1f}s] {message}"
        print(line, file=self.stream, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line+"\n")
        self.last_print = time.monotonic()

    def __call__(self, stage, completed=None, total=None, detail="", force=False):
        message = stage
        if completed is not None and total is not None:
            message += f" | {completed:,}/{total:,} ({100*completed/max(total, 1):.1f}%)"
        if detail:
            message += f" | {detail}"
        with self.lock:
            self.message = message
            if force or completed == total or time.monotonic()-self.last_print >= self.interval:
                self._emit(message)

    def stage(self, message):
        self(message, force=True)

    def _heartbeat(self):
        while not self.stopped.wait(self.interval):
            with self.lock:
                if time.monotonic()-self.last_print >= self.interval:
                    self._emit(f"working: {self.message}")

    def __enter__(self):
        self.stage("started; progress and heartbeat enabled")
        self.thread = threading.Thread(target=self._heartbeat, name=f"{self.label}-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        if exc_type is not None:
            self.stage(f"stopped: {exc_type.__name__}: {exc_value}")
