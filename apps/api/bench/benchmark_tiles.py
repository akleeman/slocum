from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RunResult:
    count: int
    p50_ms: float
    p95_ms: float
    max_ms: float


def benchmark(url: str, runs: int = 10) -> RunResult:
    durations: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        for _ in range(runs):
            start = time.perf_counter()
            response = client.get(url)
            response.raise_for_status()
            durations.append((time.perf_counter() - start) * 1000.0)
    sorted_ms = sorted(durations)
    return RunResult(
        count=runs,
        p50_ms=statistics.median(sorted_ms),
        p95_ms=sorted_ms[int(0.95 * (runs - 1))],
        max_ms=max(sorted_ms),
    )


if __name__ == "__main__":
    # Replace with a valid cycle from /meta/cycles before running.
    url = "http://127.0.0.1:8000/tiles/temperature_2m_member/2026-01-02T00:00:00/12/3/2/2"
    result = benchmark(url=url, runs=10)
    print(
        f"n={result.count} p50={result.p50_ms:.1f}ms "
        f"p95={result.p95_ms:.1f}ms max={result.max_ms:.1f}ms"
    )
