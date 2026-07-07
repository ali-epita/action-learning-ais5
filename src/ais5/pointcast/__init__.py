"""PointCast — a hands-free, on-device accessibility pointer.

Speak or type a target in plain language; PointCast grounds it to an exact pixel
on the live screen using the AIS5 LoRA-r64 Qwen2.5-VL model (85.5% on
ScreenSpot-V2), shows a high-contrast crosshair, speaks a confirmation with a
countdown, and dispatches a real click — fully offline, screen contents never
leave the machine.

This package is the Phase-3 MVP ("PointCast") described in
``AIS5_MVP_Specifications.md``. It reuses the research library's inference
contract verbatim (``ais5.prompt.format_click_prompt`` / ``parse_click`` and
``ais5.tile.crop_then_click``) so on-device behavior matches the benchmark.

Layout:
    backends/   swappable inference backends (MLX now, GGUF next), each an
                ``ais5`` ``GUIModel`` so it composes with the existing tile/eval
                machinery.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
