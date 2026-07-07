"""On-device speech-to-text via mlx_audio — fully offline.

Lazy-loads an ASR model on first use and transcribes a recorded mono float32
waveform. Best-effort: any failure returns "" so voice input never crashes the
pointer (the user can always type).

Default is Parakeet-CTC-110M: for short command phrases it transcribes in
~0.1-0.3 s (vs ~2.8 s for Whisper-large-v3-turbo), is more accurate on them,
and resides in ~0.5 GB (a gigabyte less than Whisper) so it co-fits with the
grounding VLM on 8 GB. Measured on an M1 8 GB, 2026-07-06.
"""

from __future__ import annotations

from typing import Any

from ..utils.logging import get_logger

_log = get_logger("pointcast.stt")

# Local, self-contained MLX ASR model shipped in the repo root so voice input
# runs fully offline (resolved as a local path before any HF lookup). Run from
# the repo dir, or override with --stt-model / an absolute path.
DEFAULT_STT_MODEL = "parakeet-ctc-110m-asr"


def _extract_text(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "text") and isinstance(result.text, str):
        return result.text
    if isinstance(result, (list, tuple)):
        parts = []
        for seg in result:
            t = getattr(seg, "text", None)
            if t is None and isinstance(seg, dict):
                t = seg.get("text")
            if t:
                parts.append(t)
        return " ".join(parts)
    if isinstance(result, dict):
        return result.get("text", "")
    return str(result)


class SpeechToText:
    def __init__(self, model_id: str = DEFAULT_STT_MODEL, samplerate: int = 16000, min_seconds: float = 0.3):
        self.model_id = model_id
        self.samplerate = samplerate
        self.min_seconds = min_seconds
        self._model: Any = None

    def load(self) -> SpeechToText:
        if self._model is None:
            import os

            # Offline-first guard (mirrors the MLX backend): a bare local dir
            # name must not fall through to a network HF lookup when missing.
            looks_like_hf_id = self.model_id.count("/") == 1 and not os.path.exists(self.model_id)
            if not os.path.isdir(self.model_id) and not looks_like_hf_id:
                raise FileNotFoundError(
                    f"STT model directory {self.model_id!r} not found. Run from the repo "
                    "root, or pass --stt-model with an absolute path."
                )
            from mlx_audio.stt import load as load_stt

            _log.info("loading STT model %s ...", self.model_id)
            self._model = load_stt(self.model_id)
        return self

    def transcribe(self, samples) -> str:
        """samples: numpy float32 mono waveform at ``self.samplerate``."""
        if samples is None or len(samples) < self.samplerate * self.min_seconds:
            _log.info("recording too short (%s samples) - ignoring", 0 if samples is None else len(samples))
            return ""
        try:
            self.load()
            import mlx.core as mx
            from mlx_audio.stt.generate import generate_transcription

            audio = mx.array(samples)
            result = generate_transcription(model=self._model, audio=audio, verbose=False)
            return _extract_text(result).strip()
        except Exception as e:  # noqa: BLE001
            _log.warning("transcription failed: %r", e)
            return ""
