"""LoRA training loop for grounding data (OS-Atlas + UGround mixture).

Designed to run inside a Colab/Kaggle notebook: a single `run_lora_training`
call drives data loading, optim, and checkpointing. Heavy imports are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .lora import LoRAConfig, attach_lora, count_trainable

if TYPE_CHECKING:
    from datasets import Dataset

log = get_logger(__name__)


@dataclass
class TrainingArgs:
    output_dir: str = "checkpoints/qwen2.5-vl-3b-lora"
    train_dataset: str = "OS-Copilot/OS-Atlas-data"
    train_subset_size: int = 50_000  # default below the 200K stretch goal
    eval_dataset: str | None = None
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 25
    save_steps: int = 1000
    eval_steps: int | None = None
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


def run_lora_training(
    model_name: str,
    lora: LoRAConfig,
    args: TrainingArgs,
    *,
    train_data: Dataset | None = None,
) -> Path:
    """LoRA-fine-tune `model_name` on `train_data` and write a PEFT adapter to disk.

    Returns the output directory containing `adapter_model.safetensors`.

    Notes
    -----
    The collator needs to be VLM-aware (image + text). For now this function
    delegates to HF's `Trainer` with a dataset that yields `{"input_ids", "labels",
    "pixel_values"}`. If you adapt this for ShowUI / OS-Atlas you'll likely need
    a custom collator — see `notebooks/01_task1_lora_adaptation.ipynb`.
    """
    from transformers import Trainer, TrainingArguments

    from ..models import get_model
    from ..utils.seed import set_global_seed

    set_global_seed(args.seed)

    log.info("Loading base model %s", model_name)
    base = get_model(model_name)
    if base.model is None:
        raise RuntimeError("Model wrapper did not load `self.model`")

    peft_model = attach_lora(base.model, lora)
    trainable, total = count_trainable(peft_model)
    log.info(
        "Trainable params: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / max(1, total),
    )

    if train_data is None:
        train_data = _load_default_train_data(args)

    output_dir = ensure_dir(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        report_to=["none"],
        seed=args.seed,
        **args.extra,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=base.processor,
    )
    trainer.train()
    peft_model.save_pretrained(str(output_dir))
    log.info("Saved LoRA adapter to %s", output_dir)
    return output_dir


def _load_default_train_data(args: TrainingArgs) -> Dataset:
    from datasets import load_dataset

    log.info("Streaming %s and taking %d samples", args.train_dataset, args.train_subset_size)
    ds = load_dataset(args.train_dataset, split="train", streaming=True)
    return ds.take(args.train_subset_size)
