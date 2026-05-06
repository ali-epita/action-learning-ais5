"""Quantization configurations.

We expose a thin `QuantConfig` enum-like dataclass plus factories that return
HF `BitsAndBytesConfig` (or None) ready to pass into `from_pretrained`. The
proposal asks for "8-bit if available, and W4A8 or closest reproducible
low-bit setting" — that means:

    fp16/bf16 (baseline)  → no quant
    bnb_8bit              → bitsandbytes int8 (LLM.int8)
    bnb_4bit (NF4)        → bitsandbytes 4-bit weights, bf16 compute
    awq_w4a8              → AWQ checkpoint (loaded by name; recipe is in HF Hub)

W4A8 specifically requires a model whose weights have been quantized offline
with AWQ or GPTQ. We don't run the quantization ourselves; we fetch a quantized
checkpoint and load it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantConfig:
    name: str
    kind: str  # "none" | "bnb8" | "bnb4" | "awq" | "gptq"
    bits: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_hf(self) -> Any:
        """Return the object that can be passed as `quantization_config=` to HF."""
        if self.kind in ("none",):
            return None
        if self.kind == "bnb8":
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(load_in_8bit=True, **self.extra)
        if self.kind == "bnb4":
            import torch
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.extra.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.extra.get("double_quant", True),
                bnb_4bit_compute_dtype=getattr(torch, self.extra.get("compute_dtype", "bfloat16")),
            )
        if self.kind in ("awq", "gptq"):
            # AWQ / GPTQ checkpoints embed their config in the HF repo — usually
            # nothing extra needs to be passed. The actual W4A8 recipe is
            # selected by which checkpoint you load.
            return None
        raise ValueError(f"Unknown quant kind {self.kind!r}")


def none() -> QuantConfig:
    return QuantConfig(name="fp16", kind="none")


fp16 = none


def bnb_8bit() -> QuantConfig:
    return QuantConfig(name="bnb-8bit", kind="bnb8", bits=8)


def bnb_4bit(*, quant_type: str = "nf4") -> QuantConfig:
    return QuantConfig(
        name=f"bnb-4bit-{quant_type}",
        kind="bnb4",
        bits=4,
        extra={"quant_type": quant_type, "double_quant": True, "compute_dtype": "bfloat16"},
    )


_BUILTINS = {
    "none": none,
    "fp16": none,
    "bnb8": bnb_8bit,
    "bnb-8bit": bnb_8bit,
    "bnb4": bnb_4bit,
    "bnb-4bit": bnb_4bit,
}


def resolve_quant_config(spec: str | dict | QuantConfig | None) -> QuantConfig:
    """Accept a string ('bnb8'), a dict ({'kind': 'bnb4'}), or a config — return one."""
    if spec is None:
        return none()
    if isinstance(spec, QuantConfig):
        return spec
    if isinstance(spec, str):
        if spec not in _BUILTINS:
            raise ValueError(f"Unknown quant preset {spec!r}; have {sorted(_BUILTINS)}")
        return _BUILTINS[spec]()
    if isinstance(spec, dict):
        kind = spec.get("kind") or spec.get("name") or "none"
        if kind in _BUILTINS:
            return _BUILTINS[kind](**{k: v for k, v in spec.items() if k != "kind"})
        return QuantConfig(name=spec.get("name", kind), kind=kind, extra=spec)
    raise TypeError(f"Unsupported quant spec type: {type(spec)}")
