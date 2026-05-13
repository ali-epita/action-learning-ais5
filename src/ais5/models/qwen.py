"""Qwen2.5-VL wrapper — primary generalist for Tasks 1-3."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..prompt.action import parse_click
from ..prompt.templates import format_click_prompt
from .base import ModelOutput, TorchModel


class QwenVL(TorchModel):
    family = "generalist"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 64,
        quant_config: Any = None,
        peft_adapter: str | None = None,
        **load_kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.param_count_b = _infer_param_count(model_id)
        self.max_new_tokens = max_new_tokens

        kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(torch_dtype),
            "device_map": device_map,
            **load_kwargs,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

        if peft_adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, peft_adapter)

    def predict(
        self,
        image: Image,
        instruction: str,
        *,
        prompt_template: str | None = None,
        **gen_kwargs: Any,
    ) -> ModelOutput:
        from qwen_vl_utils import process_vision_info

        prompt = format_click_prompt(
            instruction, **({"template": prompt_template} if prompt_template else {})
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
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
            metadata={"prompt": prompt, "model_id": self.model_id},
        )


def _infer_param_count(model_id: str) -> float:
    lower = model_id.lower()
    for size in ("72b", "32b", "14b", "7b", "3b", "2b"):
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
