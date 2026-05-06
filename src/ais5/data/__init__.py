"""Dataset loaders for the three benchmarks used across all tasks.

Public API:

    from ais5.data import (
        GroundingSample,
        load_screenspot_v2,
        load_screenspot_pro,
        load_osworld_g,
        load_benchmark,
    )

`load_benchmark("screenspot-v2")` is the canonical entry point so calling code
doesn't hardcode dataset names.
"""

from .osworld_g import load_osworld_g
from .registry import load_benchmark
from .screenspot import load_screenspot_pro, load_screenspot_v2
from .types import GroundingSample

__all__ = [
    "GroundingSample",
    "load_benchmark",
    "load_osworld_g",
    "load_screenspot_pro",
    "load_screenspot_v2",
]
