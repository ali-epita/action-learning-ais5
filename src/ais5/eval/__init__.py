"""Evaluation primitives shared by every modeling task."""

from .breakdown import bucket_for_area, by_target_size, by_ui_type
from .click import ClickResult, click_accuracy, point_in_bbox
from .runner import EvalRun, evaluate_model

__all__ = [
    "ClickResult",
    "EvalRun",
    "bucket_for_area",
    "by_target_size",
    "by_ui_type",
    "click_accuracy",
    "evaluate_model",
    "point_in_bbox",
]
