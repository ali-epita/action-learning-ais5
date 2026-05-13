"""PaliGemma wrapper — second generalist (Task 1 cross-backbone check)."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..prompt.action import parse_click
from .base import ModelOutput, TorchModel

# PaliGemma was pre-trained with task prefixes. For pointing tasks we use
# "point" — output is `<loc####><loc####>` indices into a 1024-bin grid.
PALIGEMMA_POINT_PROMPT = "point: {instruction}"


class PaliGemma(TorchModel):
    family = "generalist"

    def __init__(
        self,
        model_id: str = "google/paligemma-3b-mix-448",
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 32,
        quant_config: Any = None,
        peft_adapter: str | None = None,
        **load_kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.param_count_b = 3.0
        self.max_new_tokens = max_new_tokens

        kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(torch_dtype),
            "device_map": device_map,
            **load_kwargs,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, **kwargs)
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
        prompt = (prompt_template or PALIGEMMA_POINT_PROMPT).format(instruction=instruction)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            **gen_kwargs,
        )
        response: str = self.processor.decode(gen[0][input_len:], skip_special_tokens=False)

        return ModelOutput(
            text=response,
            parsed=parse_click(response, image_size=image.size),
            metadata={"prompt": prompt, "model_id": self.model_id},
        )


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            return torch.float32
        return "auto"
    return getattr(torch, name)
