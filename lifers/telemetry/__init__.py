"""
Lifers Pulse — 遥测监控系统
指标采集(Gauge/Counter/Histogram)、追踪、JSON导出
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class Metric:
    pass


class Counter(Metric):
    def __init__(self, name: str, help: str = "") -> None:
        self.name = name
        self.help = help
        self._value: int = 0

    def inc(self, delta: int = 1) -> None:
        self._value += delta

    def value(self) -> int:
        return self._value

    def reset(self) -> None:
        self._value = 0


class Gauge(Metric):
    def __init__(self, name: str, help: str = "") -> None:
        self.name = name
        self.help = help
        self._value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


class Histogram(Metric):
    def __init__(self, name: str, buckets: Optional[List[float]] = None, help: str = "") -> None:
        self.name = name
        self.help = help
        self.buckets = buckets or [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        self._count: int = 0
        self._sum: float = 0.0
        self._buckets: Dict[str, int] = {f"{b}": 0 for b in self.buckets}

    def observe(self, value: float) -> None:
        self._count += 1
        self._sum += value
        for b in self.buckets:
            if value <= b:
                self._buckets[f"{b}"] += 1

    def mean(self) -> float:
        return self._sum / max(self._count, 1)

    def value(self) -> Dict[str, Any]:
        return {"count": self._count, "sum": self._sum, "mean": self.mean(), "buckets": self._buckets}

    def reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._buckets = {f"{b}": 0 for b in self.buckets}


# ═══════════════════════════════════════════════════════════════════════════════
# Tracer (Span-based)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Span:
    id: str
    name: str
    parent_id: Optional[str] = None
    started_ms: int = 0
    ended_ms: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


class Tracer:
    def __init__(self, max_spans: int = 500) -> None:
        self._spans: Dict[str, Span] = {}
        self._active: deque = deque()
        self._completed: deque = deque(maxlen=max_spans)
        self._next_id = 0

    def start_span(self, name: str, parent_id: Optional[str] = None) -> str:
        span_id = f"span_{self._next_id}"
        self._next_id += 1
        span = Span(id=span_id, name=name, parent_id=parent_id or (self._active[-1] if self._active else None),
                    started_ms=int(time.time() * 1000))
        self._spans[span_id] = span
        self._active.append(span_id)
        return span_id

    def end_span(self, span_id: str) -> None:
        span = self._spans.get(span_id)
        if span is None:
            return
        span.ended_ms = int(time.time() * 1000)
        self._completed.append(span)
        if span_id in self._active:
            self._active.remove(span_id)

    def log(self, span_id: str, message: str, **kwargs) -> None:
        span = self._spans.get(span_id)
        if span:
            span.logs.append({"ts_ms": int(time.time() * 1000), "message": message, **kwargs})

    def set_tag(self, span_id: str, key: str, value: str) -> None:
        span = self._spans.get(span_id)
        if span:
            span.tags[key] = value

    @contextlib.contextmanager
    def span(self, name: str):
        sid = self.start_span(name)
        try:
            yield sid
        finally:
            self.end_span(sid)

    def recent_spans(self, n: int = 20) -> List[Dict]:
        spans = list(self._completed)[-n:]
        return [{"id": s.id, "name": s.name, "duration_ms": s.ended_ms - s.started_ms, "tags": s.tags} for s in spans]


import contextlib  # noqa: E402 (must be after class definition for self-referential usage)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self) -> None:
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, help: str = "") -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name, help)
        return self._counters[name]

    def gauge(self, name: str, help: str = "") -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, help)
        return self._gauges[name]

    def histogram(self, name: str, help: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, buckets, help)
        return self._histograms[name]

    def export_json(self) -> Dict[str, Any]:
        return {
            "counters": {n: c.value() for n, c in self._counters.items()},
            "gauges": {n: g.value() for n, g in self._gauges.items()},
            "histograms": {n: h.value() for n, h in self._histograms.items()},
            "exported_at_ms": int(time.time() * 1000),
        }

    def export_prometheus_text(self) -> str:
        lines = []
        for name, c in self._counters.items():
            lines.append(f"# HELP {name} {c.help}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {c.value()}")
        for name, g in self._gauges.items():
            lines.append(f"# HELP {name} {g.help}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {g.value()}")
        for name, h in self._histograms.items():
            lines.append(f"# HELP {name} {h.help}")
            lines.append(f"# TYPE {name} histogram")
            for bucket, count in h.value()["buckets"].items():
                lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            lines.append(f"{name}_count {h.value()['count']}")
            lines.append(f"{name}_sum {h.value()['sum']}")
        return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers Pulse — unified monitoring
# ═══════════════════════════════════════════════════════════════════════════════

class LifersPulse:
    """Unified telemetry: metrics + tracer + health snapshot."""

    def __init__(self) -> None:
        self.metrics = MetricsRegistry()
        self.tracer = Tracer()
        self._startup_ms = int(time.time() * 1000)
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        self.metrics.counter("lifers_turns_total", "Total dialogue turns processed")
        self.metrics.gauge("lifers_memory_items", "Items in long-term memory")
        self.metrics.gauge("lifers_vector_items", "Items in vector store")
        self.metrics.histogram("lifers_inference_latency_ms", "Inference latency in ms",
                               [100, 500, 1000, 5000, 15000, 30000])
        self.metrics.histogram("lifers_tool_latency_ms", "Tool execution latency",
                               [10, 50, 100, 500, 1000, 5000])
        self.metrics.counter("lifers_errors_total", "Total errors encountered")
        self.metrics.gauge("lifers_uptime_sec", "Process uptime in seconds")

    def tick(self) -> Dict[str, Any]:
        """Called periodically to update gauges and return snapshot."""
        uptime = (int(time.time() * 1000) - self._startup_ms) / 1000
        self.metrics.gauge("lifers_uptime_sec").set(uptime)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.export_json(),
            "active_spans": len(self.tracer._active),
            "completed_spans": len(self.tracer._completed),
            "uptime_sec": (int(time.time() * 1000) - self._startup_ms) / 1000,
            "ts_ms": int(time.time() * 1000),
        }

    def export_metrics_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)
