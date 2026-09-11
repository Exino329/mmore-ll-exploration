"""Measuring what a pipeline stage costs.

Each stage runs as its own subprocess. That is deliberate: peak memory is only meaningful
per stage, embedding models and Milvus handles must not stay warm from a previous stage,
and it is exactly how the stages are invoked in practice (``mmore index``,
``mmore graph-index``).
"""

import json
import os
import re
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import psutil

_SAMPLE_INTERVAL = 0.1
"""Peak RSS is sampled, not traced: a spike shorter than this can be missed."""


@dataclass
class StageTiming:
    name: str
    command: List[str]
    wall_seconds: float
    cpu_seconds: float
    peak_rss_mb: float
    returncode: int
    log_path: str
    extra: Dict[str, object] = field(default_factory=dict)


def _tree_rss(process: psutil.Process) -> int:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_stage(
    name: str,
    command: List[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> StageTiming:
    """Run ``command``, returning its wall time, child CPU time and peak tree RSS."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **(env or {})}

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=environment,
        )
        monitor = psutil.Process(process.pid)
        peak = 0
        while process.poll() is None:
            try:
                peak = max(peak, _tree_rss(monitor))
            except psutil.NoSuchProcess:
                break
            time.sleep(_SAMPLE_INTERVAL)
        returncode = process.wait()

    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)

    return StageTiming(
        name=name,
        command=command,
        wall_seconds=wall,
        cpu_seconds=cpu,
        peak_rss_mb=peak / (1024**2),
        returncode=returncode,
        log_path=str(log_path),
    )


def mmore_command(stage: str, *args: str) -> List[str]:
    """``mmore`` invoked through the current interpreter, so the venv is inherited."""
    return [sys.executable, "-m", "mmore", stage, *args]


_REPORTED_SECONDS = re.compile(r"done in ([\d.]+)s")


def reported_seconds(log_path: Path) -> Optional[float]:
    """The stage's own timing, taken from the summary card it prints.

    mmore subtracts model-loading time from that number, while the wall clock above
    includes it. On a small corpus the two differ by more than the work itself: loading
    the embedding model and starting Milvus is a fixed cost of ten-odd seconds.
    """
    if not log_path.exists():
        return None
    matches = _REPORTED_SECONDS.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    return float(matches[-1]) if matches else None


def path_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024**2)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


def write_timings(timings: List[StageTiming], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(t) for t in timings], indent=2), encoding="utf-8"
    )
