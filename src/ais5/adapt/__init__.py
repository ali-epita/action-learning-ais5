"""Parameter-efficient adaptation (Task 1)."""

from .data import (
    ROW_ADAPTERS,
    GroundingTrainExample,
    PaliGemmaGroundingCollator,
    QwenVLGroundingCollator,
    adapt_auto_row,
    adapt_os_atlas_row,
    adapt_uground_row,
    make_collator,
    stream_grounding_examples,
)
from .lora import LoRAConfig, attach_lora, count_trainable
from .os_atlas import available_subsets, os_atlas_dataset, stream_os_atlas
from .train import TrainingArgs, run_lora_training

__all__ = [
    "ROW_ADAPTERS",
    "GroundingTrainExample",
    "LoRAConfig",
    "PaliGemmaGroundingCollator",
    "QwenVLGroundingCollator",
    "TrainingArgs",
    "adapt_auto_row",
    "adapt_os_atlas_row",
    "adapt_uground_row",
    "attach_lora",
    "available_subsets",
    "count_trainable",
    "make_collator",
    "os_atlas_dataset",
    "run_lora_training",
    "stream_grounding_examples",
    "stream_os_atlas",
]
