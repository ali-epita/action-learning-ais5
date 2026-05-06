"""Model wrappers behind a unified `GUIModel` interface.

    from ais5.models import get_model
    model = get_model("qwen2.5-vl-3b")
    out = model.predict(image, "click the save button")

The registry below is the single source of truth — adding a new model means
adding one entry here, not threading another import path through every notebook.
"""

from .base import GUIModel, ModelOutput
from .registry import ModelSpec, get_model, list_models, register

__all__ = [
    "GUIModel",
    "ModelOutput",
    "ModelSpec",
    "get_model",
    "list_models",
    "register",
]
