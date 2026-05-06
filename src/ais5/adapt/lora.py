"""Thin wrapper around PEFT's LoRA so Task 1 can sweep ranks declaratively."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import PeftModel
    from torch.nn import Module


# Sensible default target modules per backbone family. Override per-config when
# a checkpoint uses non-standard projection names.
DEFAULT_TARGET_MODULES = {
    "qwen2.5-vl": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "paligemma": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


@dataclass
class LoRAConfig:
    """All knobs needed to instantiate a peft.LoraConfig.

    Defaults match the proposal's rank sweep starting point (r=8, alpha=16).
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)
    backbone: str = "qwen2.5-vl"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] = field(default_factory=list)

    def resolved_targets(self) -> list[str]:
        if self.target_modules:
            return list(self.target_modules)
        return list(DEFAULT_TARGET_MODULES.get(self.backbone, ["q_proj", "v_proj"]))

    def to_peft(self) -> Any:
        from peft import LoraConfig

        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            target_modules=self.resolved_targets(),
            task_type=self.task_type,
            modules_to_save=self.modules_to_save or None,
        )


def attach_lora(model: Module, cfg: LoRAConfig) -> PeftModel:
    """Wrap `model` with `LoraConfig` adapters and return the PeftModel."""
    from peft import get_peft_model

    return get_peft_model(model, cfg.to_peft())


def count_trainable(model: Module) -> tuple[int, int]:
    """Return `(trainable, total)` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
