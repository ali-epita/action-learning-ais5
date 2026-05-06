"""ais5 — Small VLMs vs. GUI Specialists.

Top-level package for the AIS 5 Action Learning project.

The library is organized so each modeling task imports a thin slice:

    from ais5.data import load_screenspot_v2
    from ais5.eval import click_accuracy
    from ais5.models import get_model
    from ais5.adapt import LoRAConfig
    from ais5.tile import crop_then_click
    from ais5.quant import QuantConfig
    from ais5.bench import benchmark
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
