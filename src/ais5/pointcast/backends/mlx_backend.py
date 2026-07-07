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
from ...prompt.action import ParsedAction, parse_click
from ...prompt.templates import CLICK_PROMPT_NORM1000, format_click_prompt
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
        self._pil_supported: bool | None = None  # None = probe on first call
        # Coordinate profile, set at load() from the model's config.json:
        #   "absolute": image-pixel outputs (Qwen2.5-VL family incl. our r64)
        #   "norm1000": [0,1000]-normalized outputs (Qwen3-VL family)
        self.coord_profile = "absolute"
        self.last_result: Any = None  # most recent GenerationResult (timing/mem)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def load(self) -> MLXBackend:
        if self._model is not None:
            return self
        import os

        # Offline-first guard: a bare name like "mlx-qwen-r64-4bit" is a LOCAL
        # directory. If it is missing, fail fast with a clear message instead of
        # letting mlx_vlm treat it as a HF repo id and hit the network.
        looks_like_hf_id = self.model_path.count("/") == 1 and not os.path.exists(self.model_path)
        if not os.path.isdir(self.model_path) and not looks_like_hf_id:
            raise FileNotFoundError(
                f"model directory {self.model_path!r} not found. PointCast is offline-first: "
                "run from the repo root, or pass --model with an absolute path to the "
                "mlx-qwen-r64-4bit directory."
            )
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
        self.coord_profile = _coord_profile_for(self._config)
        self.name = os.path.basename(os.path.normpath(self.model_path)) or self.name
        _log.info("loaded %s (coordinate profile: %s)", self.name, self.coord_profile)
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
        # Fallback path when the installed mlx_vlm rejects in-memory PIL images:
        # write a fresh file per call so nothing caches a stale frame by path.
        self._frame_counter += 1
        path = os.path.join(self._tmpdir, f"frame_{self._frame_counter}.png")
        image.convert("RGB").save(path)
        return path

    def _debug_dump(self, img: Image) -> None:
        """When AIS5_DEBUG_FRAMES is set to a directory, save the EXACT image
        each grounding/ask call sees, so we can confirm what the model actually
        grounded on (e.g. whether a PointCast overlay leaked into the frame)."""
        d = os.environ.get("AIS5_DEBUG_FRAMES")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            self._frame_counter += 1
            p = os.path.join(d, f"ground_{self._frame_counter:03d}.png")
            img.convert("RGB").save(p)
            _log.info("    [debug] grounding frame saved -> %s", p)
        except Exception:  # noqa: BLE001
            pass

    def _run_generate(self, formatted: str, img: Image, gen_kwargs: dict[str, Any]) -> Any:
        """Call mlx_vlm.generate with the PIL image directly (no PNG round-trip
        through disk, ~100-200 ms per call); fall back to a temp file once if
        this mlx_vlm build only accepts paths, and clean the file up after."""
        self._debug_dump(img)
        from mlx_vlm import generate

        # The reclaim runs in a finally so a FAILED generation (e.g. a Metal
        # OOM mid-call) releases its buffers too: skipping it would keep the
        # retry under the very cache pressure that broke this call.
        try:
            if self._pil_supported is not False:
                try:
                    res = generate(self._model, self._processor, formatted, image=[img], **gen_kwargs)
                    self._pil_supported = True
                    return res
                except (TypeError, ValueError, AttributeError, OSError) as e:
                    if self._pil_supported is True:
                        raise  # PIL images worked before: this is a real inference error
                    _log.info("mlx_vlm rejected an in-memory image (%r); using temp files", e)
                    self._pil_supported = False
            path = self._image_to_path(img)
            try:
                return generate(self._model, self._processor, formatted, image=[path], **gen_kwargs)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        finally:
            self._reclaim()

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
        from mlx_vlm.prompt_utils import apply_chat_template

        img = image.convert("RGB")
        template = CLICK_PROMPT_NORM1000 if self.coord_profile == "norm1000" else None
        prompt = format_click_prompt(instruction, **({"template": template} if template else {}))
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
        res = self._run_generate(formatted, img, gen_kwargs)
        dt = time.perf_counter() - t0
        self.last_result = res
        text = getattr(res, "text", str(res))
        parsed = _to_image_pixels(parse_click(text, image_size=img.size), img.size, self.coord_profile)
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
        return ModelOutput(text=text, parsed=parsed, metadata=meta)

    def ask(self, image: Image, prompt: str, *, max_tokens: int = 128) -> str:
        """Free-form VQA: send ``prompt`` as-is (no click wrapper), return text."""
        if self._model is None:
            self.load()
        from mlx_vlm.prompt_utils import apply_chat_template

        img = image.convert("RGB")
        formatted = apply_chat_template(self._processor, self._config, prompt, num_images=1)
        t0 = time.perf_counter()
        res = self._run_generate(
            formatted, img, {"max_tokens": max_tokens, "temperature": 0.0, "verbose": False}
        )
        text = getattr(res, "text", str(res)) or ""
        self.last_result = res
        _log.info("    ask (%dx%d): %.1fs -> %r", img.size[0], img.size[1], time.perf_counter() - t0, text[:80])
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


def _coord_profile_for(config: Any) -> str:
    """Coordinate convention from the model config's ``model_type``.

    Qwen3-VL grounds in [0,1000]-normalized coordinates; Qwen2.5-VL (and our
    r64 fine-tune, trained on image pixels) grounds in absolute pixels.
    """
    mt = ""
    try:
        if isinstance(config, dict):
            mt = str(config.get("model_type", ""))
        else:
            mt = str(getattr(config, "model_type", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return "norm1000" if mt.startswith("qwen3_vl") else "absolute"


def _to_image_pixels(parsed: ParsedAction, size: tuple[int, int], profile: str) -> ParsedAction:
    """Convert a parsed click from the model's native coordinate space into
    image pixels (the contract every caller expects)."""
    if profile != "norm1000" or parsed.point is None:
        return parsed
    from dataclasses import replace

    w, h = size
    x, y = parsed.point
    point = (x / 1000.0 * w, y / 1000.0 * h)
    bbox = parsed.bbox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bbox = (x1 / 1000.0 * w, y1 / 1000.0 * h, x2 / 1000.0 * w, y2 / 1000.0 * h)
    return replace(parsed, point=point, bbox=bbox)
