"""Model registry. Spec dataclass + a name → factory map."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .base import GUIModel


@dataclass
class ModelSpec:
    name: str
    family: str  # "generalist" | "specialist"
    param_count_b: float
    hf_id: str
    factory: Callable[..., GUIModel]
    extra: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Model {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def list_models() -> list[ModelSpec]:
    return list(_REGISTRY.values())


def get_model(name: str, **kwargs: Any) -> GUIModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(sorted(_REGISTRY))}")
    spec = _REGISTRY[name]
    overrides = {**spec.extra, **kwargs}
    return spec.factory(model_id=spec.hf_id, **overrides)


# ── built-in registrations (deferred imports keep heavy deps lazy) ────────────


def _register_defaults() -> None:
    from .paligemma import PaliGemma
    from .qwen import QwenVL
    from .specialists import FerretUILite, OSAtlas, ShowUI

    register(
        ModelSpec(
            name="qwen2.5-vl-3b",
            family="generalist",
            param_count_b=3.0,
            hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
            factory=QwenVL,
        )
    )
    register(
        ModelSpec(
            name="qwen2.5-vl-7b",
            family="generalist",
            param_count_b=7.0,
            hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
            factory=QwenVL,
        )
    )
    register(
        ModelSpec(
            name="paligemma-3b",
            family="generalist",
            param_count_b=3.0,
            hf_id="google/paligemma-3b-mix-448",
            factory=PaliGemma,
        )
    )
    register(
        ModelSpec(
            name="os-atlas-4b",
            family="specialist",
            param_count_b=4.0,
            hf_id="OS-Copilot/OS-Atlas-Pro-4B",
            factory=OSAtlas,
        )
    )
    register(
        ModelSpec(
            name="showui-2b",
            family="specialist",
            param_count_b=2.0,
            hf_id="showlab/ShowUI-2B",
            factory=ShowUI,
        )
    )
    register(
        ModelSpec(
            name="ferret-ui-lite-3b",
            family="specialist",
            param_count_b=3.0,
            hf_id="apple/Ferret-UI-Lite-3B",  # placeholder; verify on release
            factory=FerretUILite,
        )
    )


_register_defaults()
