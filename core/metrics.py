"""
Centralized performance metrics collector.
Tracks per-stage latencies and provides rolling statistics for the pipeline.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StageMetrics:
    name: str
    _times: deque = field(default_factory=lambda: deque(maxlen=120))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def last_ms(self) -> float:
        with self._lock:
            return self._times[-1] * 1000 if self._times else 0.0

    @property
    def avg_ms(self) -> float:
        with self._lock:
            if not self._times:
                return 0.0
            return (sum(self._times) / len(self._times)) * 1000

    @property
    def history_ms(self) -> list[float]:
        with self._lock:
            return [t * 1000 for t in self._times]

    def record(self, duration_sec: float):
        with self._lock:
            self._times.append(duration_sec)


class PipelineMetrics:
    """Collects timing data for all pipeline stages."""

    def __init__(self):
        self._stages: dict[str, StageMetrics] = {}
        self._fps_times: deque = deque(maxlen=120)
        self._lock = threading.Lock()

    def stage(self, name: str) -> StageMetrics:
        if name not in self._stages:
            self._stages[name] = StageMetrics(name=name)
        return self._stages[name]

    def tick_fps(self):
        with self._lock:
            self._fps_times.append(time.perf_counter())

    @property
    def pipeline_fps(self) -> float:
        with self._lock:
            if len(self._fps_times) < 2:
                return 0.0
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._fps_times) - 1) / elapsed

    def snapshot(self) -> dict:
        return {
            "pipeline_fps": round(self.pipeline_fps, 1),
            "stages": {
                name: {"last_ms": round(s.last_ms, 1), "avg_ms": round(s.avg_ms, 1)}
                for name, s in self._stages.items()
            },
        }


class Timer:
    """Context manager for timing a code block and recording to a StageMetrics."""

    def __init__(self, stage: StageMetrics):
        self._stage = stage
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._stage.record(time.perf_counter() - self._start)
