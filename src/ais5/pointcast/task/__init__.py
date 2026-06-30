"""Multi-step task mode for PointCast (Stage 1, 8 GB-friendly).

Turns a spoken/typed goal ("how much free storage do I have") into a short,
deterministic sequence of grounded clicks (Settings -> General -> Storage),
then reads the answer off the final screen with the same on-device model and
speaks it. The procedure comes from a curated recipe (or a macOS deep-link),
not from a runtime planner, so it stays reliable and fits the 3B model.
"""

from .intent import match_recipe
from .recipes import RECIPES, Recipe, Step
from .runner import TaskCallbacks, TaskRunner

__all__ = ["RECIPES", "Recipe", "Step", "match_recipe", "TaskRunner", "TaskCallbacks"]
