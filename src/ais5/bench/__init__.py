"""Efficiency benchmarking + Pareto fronts (Task 3)."""

from .latency import LatencyStats, measure_latency
from .memory import MemoryStats, measure_peak_memory, vram_budget_check
from .pareto import ParetoPoint, pareto_front
from .profile import BenchResult, ComponentTimings, find_vision_module, run_full_benchmark

__all__ = [
    "BenchResult",
    "ComponentTimings",
    "LatencyStats",
    "MemoryStats",
    "ParetoPoint",
    "find_vision_module",
    "measure_latency",
    "measure_peak_memory",
    "pareto_front",
    "run_full_benchmark",
    "vram_budget_check",
]
