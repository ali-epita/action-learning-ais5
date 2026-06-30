"""MLX (Apple Silicon) backend — the on-device default.

Loads the 4-bit ``mlx-qwen-r64-4bit`` artifact via ``mlx_vlm`` and runs the
exact same prompt + click parser as the benchmark, so behavior matches the
85.5% ScreenSpot-V2 result. Greedy decoding (``temperature=0``) is the
benchmarked path; stochastic decoding feeds disambiguation.

Requires the ``.venv-demo`` environment (mlx, mlx_vlm). Heavy imports are done
lazily so importing this module on a non-Mac doesn't fail.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

from PIL.Image import Image

from ...models.base import ModelOutput
from ...prompt.action import parse_click
from ...prompt.templates import format_click_prompt
from ...utils.logging import get_logger
from .base import DEFAULT_MAX_TOKENS, GroundingBackend

DEFAULT_MODEL_PATH = "mlx-qwen-r64-4bit"
DEFAULT_CACHE_LIMIT_MB = 384  # cap MLX's reusable buffer cache (8 GB-friendly)
_log = get_logger("pointcast.mlx")


class MLXBackend(GroundingBackend):
    """Qwen2.5-VL-3B + LoRA-r64, 4-bit MLX, fully on-device."""

    backend_id = "mlx"
    name = "qwen2.5-vl-3b-r64-mlx-4bit"
    param_count_b = 3.0
    family = "generalist"

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, *, max_tokens: int = DEFAULT_MAX_TOKENS,
                 cache_limit_mb: int = DEFAULT_CACHE_LIMIT_MB):
        self.model_path = model_path
        self.default_max_tokens = max_tokens
        self.cache_limit_mb = cache_limit_mb
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None
        self._tmpdir = tempfile.mkdtemp(prefix="pointcast_mlx_")
        self._frame_counter = 0
        self.last_result: Any = None  # most recent GenerationResult (timing/mem)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def load(self) -> MLXBackend:
        if self._model is not None:
            return self
        from mlx_vlm import load

        self._model, self._processor = load(self.model_path)
        try:
            self._config = self._model.config
        except Exception:
            from mlx_vlm.utils import load_config

            self._config = load_config(self.model_path)
        try:  # bound MLX's reusable buffer cache so it cannot grow unbounded on 8 GB
            import mlx.core as mx

            mx.set_cache_limit(self.cache_limit_mb * 1024 * 1024)
        except Exception:  # noqa: BLE001
            pass
        return self

    def _reclaim(self) -> None:
        """Return MLX's freed buffers to the OS after each call so memory does not
        creep up across the many calls in a task (the cause of the Metal OOM on
        8 GB, especially while screen-sharing)."""
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── inference ────────────────────────────────────────────────────────────
    def _image_to_path(self, image: Image) -> str:
        # mlx_vlm.generate consumes image *file paths*; write a fresh file per
        # call so nothing caches a stale frame by path.
        self._frame_counter += 1
        path = os.path.join(self._tmpdir, f"frame_{self._frame_counter}.png")
        image.convert("RGB").save(path)
        return path

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
        if self._model is None:
            self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        img = image.convert("RGB")
        path = self._image_to_path(img)
        prompt = format_click_prompt(instruction)
        formatted = apply_chat_template(self._processor, self._config, prompt, num_images=1)

        gen_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens or self.default_max_tokens,
            "verbose": False,
        }
        if sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        else:
            gen_kwargs["temperature"] = 0.0  # greedy → benchmark parity

        t0 = time.perf_counter()
        res = generate(self._model, self._processor, formatted, image=[path], **gen_kwargs)
        dt = time.perf_counter() - t0
        self.last_result = res
        text = getattr(res, "text", str(res))
        parsed = parse_click(text, image_size=img.size)
        mode = "sampled" if sample else "greedy"
        _log.info(
            "    model call (%s, %dx%d): %.1fs, %s tok -> %r",
            mode, img.size[0], img.size[1], dt,
            getattr(res, "generation_tokens", "?"), (text or "")[:48],
        )

        meta: dict[str, Any] = {"backend": self.backend_id, "sampled": sample}
        for attr in ("prompt_tokens", "generation_tokens", "total_tokens", "generation_tps", "peak_memory"):
            val = getattr(res, attr, None)
            if val is not None:
                meta[attr] = val
        self._reclaim()
        return ModelOutput(text=text, parsed=parsed, metadata=meta)

    def ask(self, image: Image, prompt: str, *, max_tokens: int = 128) -> str:
        """Free-form VQA: send ``prompt`` as-is (no click wrapper), return text."""
        if self._model is None:
            self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        img = image.convert("RGB")
        path = self._image_to_path(img)
        formatted = apply_chat_template(self._processor, self._config, prompt, num_images=1)
        t0 = time.perf_counter()
        res = generate(
            self._model, self._processor, formatted, image=[path],
            max_tokens=max_tokens, temperature=0.0, verbose=False,
        )
        text = getattr(res, "text", str(res)) or ""
        self.last_result = res
        _log.info("    ask (%dx%d): %.1fs -> %r", img.size[0], img.size[1], time.perf_counter() - t0, text[:80])
        self._reclaim()
        return text.strip()

    # ── trust-panel hooks ────────────────────────────────────────────────────
    def peak_memory_bytes(self) -> int | None:
        try:
            import mlx.core as mx

            return int(mx.get_peak_memory())
        except Exception:
            # Fall back to the last GenerationResult's peak_memory (reported in GB).
            pm = getattr(self.last_result, "peak_memory", None)
            return int(pm * 1e9) if pm else None

    def reset_peak_memory(self) -> None:
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except Exception:
            pass
