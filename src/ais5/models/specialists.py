"""GUI-specialist wrappers: OS-Atlas, ShowUI, Ferret-UI Lite.

These are baselines (no fine-tuning by us). All three were trained as
Qwen2-VL / Qwen2.5-VL derivatives, so they share the chat-template + image
input conventions and reuse the Qwen wrapper's predict path. Each subclass
overrides only what differs (prompt template, action format).

Ferret-UI Lite checkpoint availability is not yet confirmed at the time of
writing; the spec is a placeholder until Apple publishes weights.
"""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..prompt.action import parse_click
from ..prompt.templates import format_click_prompt
from .base import ModelOutput, TorchModel


class _QwenLikeSpecialist(TorchModel):
    """Shared loading + predict for Qwen2-VL-derived specialists."""

    family = "specialist"
    default_prompt: str | None = None

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 64,
        quant_config: Any = None,
        **load_kwargs: Any,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.param_count_b = _infer_param_count(model_id)
        self.max_new_tokens = max_new_tokens

        kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(torch_dtype),
            "device_map": device_map,
            "trust_remote_code": True,
            **load_kwargs,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        # Use AutoModel because each specialist sometimes ships its own class.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _build_messages(self, image: Image, instruction: str) -> list[dict]:
        prompt = format_click_prompt(
            instruction,
            **({"template": self.default_prompt} if self.default_prompt else {}),
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def predict(
        self,
        image: Image,
        instruction: str,
        **gen_kwargs: Any,
    ) -> ModelOutput:
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(image, instruction)
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            **gen_kwargs,
        )
        generated = gen[:, inputs.input_ids.shape[1] :]
        response: str = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return ModelOutput(
            text=response,
            parsed=parse_click(response, image_size=image.size),
            metadata={"prompt": chat_text, "model_id": self.model_id},
        )


class OSAtlas(_QwenLikeSpecialist):
    """OS-Atlas (paper #2). Outputs `(x, y)` pixel tuples for click actions."""

    default_prompt = (
        "In this UI screenshot, what is the position of the element corresponding "
        'to the command "{instruction}" (with bbox)?'
    )


class ShowUI(_QwenLikeSpecialist):
    """ShowUI-2B (paper #3). Smallest specialist in the comparison."""

    default_prompt = (
        'Click on the UI element that matches: "{instruction}". '
        "Reply with `{{x: <int>, y: <int>}}` in screen pixels."
    )


class FerretUILite(_QwenLikeSpecialist):
    """Ferret-UI Lite-3B (paper #1) — primary specialist target.

    NOTE: As of the project proposal, weights may not yet be on the Hub. The
    `model_id` in the registry is a placeholder; update it once Apple releases.
    Until then, evaluating this row will fail at load time — that's the
    documented "Risk" branch in Task 3's backup plan.
    """

    default_prompt = (
        "You are a mobile UI agent. Locate the element described and click it. "
        "Output `<click>x, y</click>` in screen pixels.\n\nElement: {instruction}"
    )


def _infer_param_count(model_id: str) -> float:
    lower = model_id.lower()
    for size in ("8b", "7b", "4b", "3b", "2b", "1b"):
        if size in lower:
            return float(size.rstrip("b"))
    return 0.0


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            return torch.float32
        return "auto"
    return getattr(torch, name)
