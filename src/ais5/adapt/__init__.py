"""Parameter-efficient adaptation (Task 1)."""

from .data import (
    GroundingTrainExample,
    PaliGemmaGroundingCollator,
    QwenVLGroundingCollator,
    ROW_ADAPTERS,
    adapt_auto_row,
    adapt_os_atlas_row,
    adapt_uground_row,
    make_collator,
    stream_grounding_examples,
)
from .lora import LoRAConfig, attach_lora, count_trainable
from .train import TrainingArgs, run_lora_training

__all__ = [
    "GroundingTrainExample",
    "LoRAConfig",
    "PaliGemmaGroundingCollator",
    "QwenVLGroundingCollator",
    "ROW_ADAPTERS",
    "TrainingArgs",
    "adapt_auto_row",
    "adapt_os_atlas_row",
    "adapt_uground_row",
    "attach_lora",
    "count_trainable",
    "make_collator",
    "run_lora_training",
    "stream_grounding_examples",
]
