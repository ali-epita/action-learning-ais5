"""Backend interface shared by every PointCast inference engine.

A backend is an ``ais5`` ``GUIModel`` — so it composes with
``ais5.tile.crop_then_click`` and ``ais5.eval`` unchanged — plus a stable
``backend_id`` and a few optional resource hooks the trust panel reads
(peak memory) and that the Try-Twice harness uses (stochastic sampling).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from PIL.Image import Image

from ...models.base import GUIModel, ModelOutput

DEFAULT_MAX_TOKENS = 64  # matches the benchmarked generalist setting


class GroundingBackend(GUIModel):
    """Common base for swappable grounding backends.

    Subclasses set ``backend_id``, ``name``, ``param_count_b`` and implement
    ``predict``. ``predict`` must accept the standard greedy call
    (``sample=False`` → deterministic, benchmark-parity) and a stochastic call
    (``sample=True`` with ``temperature``/``top_p``) used to generate diverse
    candidates for disambiguation.
    """

    backend_id: str = "base"
    family: str = "generalist"

    @abstractmethod
    def predict(
        self,
        image: Image,
        instruction: str,
        *,
        sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Ground ``instruction`` in ``image`` → ``ModelOutput`` whose
        ``.parsed.point`` is in image-pixel coordinates."""
        ...

    # ── free-form VQA / planning (optional) ──────────────────────────────────
    def ask(self, image: Image, prompt: str, *, max_tokens: int = 128) -> str:
        """Free-form question about ``image`` → raw answer text.

        Unlike ``predict`` this sends ``prompt`` verbatim (no click-prompt
        wrapper) and does no click parsing, so the same on-device model can read
        a value off the screen (e.g. "how much free storage is shown?"). Backends
        that cannot do this should leave the default, which raises.
        """
        raise NotImplementedError(f"{self.backend_id} backend does not support ask()")

    # ── lifecycle ────────────────────────────────────────────────────────────
    def load(self) -> GroundingBackend:
        """Eagerly load weights (otherwise loaded lazily on first predict)."""
        return self

    def close(self) -> None:
        """Release resources (subprocesses, temp files)."""

    # ── trust-panel / profiling hooks (optional) ─────────────────────────────
    def peak_memory_bytes(self) -> int | None:
        """Peak resident model memory in bytes, if the backend can report it."""
        return None

    def reset_peak_memory(self) -> None:
        """Reset the peak-memory counter (call before a fresh measurement)."""
