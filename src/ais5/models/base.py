"""Common interface every model wrapper conforms to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL.Image import Image

from ..prompt.action import ParsedAction


@dataclass
class ModelOutput:
    text: str
    parsed: ParsedAction
    metadata: dict[str, Any] = field(default_factory=dict)


class GUIModel(ABC):
    """Anything that can take (screenshot, instruction) → (point, raw_text).

    Concrete subclasses must set `name` and `param_count_b`, and implement
    `predict`. `to(device)` is provided by `TorchModel` for HF-backed models.
    """

    name: str
    param_count_b: float
    family: str = "unknown"  # "generalist" | "specialist"

    @abstractmethod
    def predict(self, image: Image, instruction: str, **kwargs: Any) -> ModelOutput: ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"params={self.param_count_b}B, family={self.family!r})"
        )


class TorchModel(GUIModel):
    """Mixin for HF-backed models. Subclasses populate `self.model` + `self.processor`."""

    model: Any = None
    processor: Any = None

    def to(self, device: str) -> TorchModel:
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def num_parameters(self) -> int | None:
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters())

    def num_trainable_parameters(self) -> int | None:
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
