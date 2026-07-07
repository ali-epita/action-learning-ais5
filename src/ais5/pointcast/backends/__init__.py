"""Swappable inference backends for PointCast.

Every backend conforms to ``ais5.models.base.GUIModel`` (via
``GroundingBackend``), so it drops straight into ``ais5.tile.crop_then_click``
and the eval harness with no adapter code. MLX is the on-device default; GGUF
(via llama-server) is the switchable cross-platform backend.
"""

from __future__ import annotations

from .base import GroundingBackend

__all__ = ["GroundingBackend", "get_backend"]


def get_backend(kind: str = "mlx", **kwargs) -> GroundingBackend:
    """Construct a backend by name. Heavy deps are imported lazily inside each
    backend, so importing this module stays cheap.

    kind: "mlx" (Apple Silicon, default) | "gguf" (llama-server, added in M6).
    """
    kind = kind.lower()
    if kind == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend(**kwargs)
    if kind == "gguf":
        try:
            from .gguf_backend import GGUFBackend  # added in milestone M6
        except ImportError as e:
            raise NotImplementedError(
                "the GGUF backend is milestone M6 and is not implemented yet — "
                "use --backend mlx (the GGUF artifacts run standalone via "
                "llama.cpp; see MODEL_DEPLOY.md)"
            ) from e

        return GGUFBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r} (expected 'mlx' or 'gguf')")
