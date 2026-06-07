"""LoRA training loop for grounding data.

Designed to run inside a Colab/Kaggle notebook: a single `run_lora_training`
call drives data loading, optim, and checkpointing. Heavy imports are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .data import make_collator
from .lora import LoRAConfig, attach_lora, count_trainable

if TYPE_CHECKING:
    from datasets import Dataset

log = get_logger(__name__)


@dataclass
class TrainingArgs:
    output_dir: str = "checkpoints/qwen2.5-vl-3b-lora"
    train_dataset: str = "OS-Copilot/OS-Atlas-data"
    train_subset_size: int = 50_000
    adapter: str = "auto"  # key in ROW_ADAPTERS, or use a custom callable via train_data
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
    save_total_limit: int | None = 1  # cap on-disk checkpoints (None = keep all)
    gradient_checkpointing: bool = False
    eval_steps: int | None = None
    os_atlas_subsets: tuple[str, ...] | None = None  # only used for OS-Atlas-data
    report_to: list[str] | None = None  # e.g. ["wandb"]; None -> no tracking
    run_name: str | None = None  # tracker run name (W&B etc.)
    skip_bad_vision_batches: bool = True
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


def run_lora_training(
    model_name: str,
    lora: LoRAConfig,
    args: TrainingArgs,
    *,
    train_data: Any = None,
) -> Path:
    """LoRA-fine-tune `model_name` on `train_data` and write a PEFT adapter to disk.

    Returns the output directory containing `adapter_model.safetensors`. Pass
    a custom `train_data` (HF `Dataset` of dict rows) to override `args.train_dataset`.
    """
    from transformers import Trainer, TrainingArguments

    from ..models import get_model
    from ..utils.seed import set_global_seed

    set_global_seed(args.seed)

    log.info("Loading base model %s", model_name)
    base = get_model(model_name)
    if base.model is None or base.processor is None:
        raise RuntimeError(
            f"{model_name} wrapper did not populate .model / .processor"
        )

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

    collator = make_collator(
        backbone=lora.backbone,
        processor=base.processor,
        adapter=args.adapter,
    )

    output_dir = ensure_dir(args.output_dir)
    streaming = not hasattr(train_data, "__len__")
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "report_to": args.report_to if args.report_to is not None else ["none"],
        "run_name": args.run_name,
        "seed": args.seed,
        "remove_unused_columns": False,  # the collator needs raw dataset rows
        **args.extra,
    }
    if streaming:
        # IterableDataset has no length; Trainer needs max_steps in that case.
        effective_batch = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        training_kwargs["max_steps"] = max(
            1,
            int(args.train_subset_size * args.num_train_epochs / effective_batch),
        )
    else:
        training_kwargs["num_train_epochs"] = args.num_train_epochs

    training_args = TrainingArguments(**training_kwargs)

    trainer_cls = _skip_bad_vision_trainer(Trainer) if args.skip_bad_vision_batches else Trainer
    trainer = trainer_cls(
        model=peft_model,
        args=training_args,
        train_dataset=train_data,
        data_collator=collator,
        tokenizer=base.processor,
    )
    trainer.train()
    peft_model.save_pretrained(str(output_dir))
    log.info("Saved LoRA adapter to %s", output_dir)

    out_path = Path(output_dir)
    # Free the training model's GPU memory before returning so the next load
    # (eval, or the next rank) gets the whole device instead of offloading
    # layers to CPU, which makes inference crawl.
    from ..utils.env import free_model

    del trainer, peft_model, base
    free_model()
    return out_path


def _load_default_train_data(args: TrainingArgs) -> Dataset:
    """Load `train_subset_size` rows from `args.train_dataset`.

    OS-Atlas-data ships labels as separate per-domain JSONs that `load_dataset`
    cannot pair to the image zips, so it routes through the custom
    `os_atlas_dataset` loader. Everything else streams via HF: `.take(N)` fetches
    rows on demand and caches nothing, returning an IterableDataset so
    `run_lora_training` computes `max_steps` instead of using num_train_epochs.
    """
    if args.train_dataset == "OS-Copilot/OS-Atlas-data":
        from .os_atlas import DEFAULT_SUBSETS, os_atlas_dataset

        subsets = args.os_atlas_subsets or DEFAULT_SUBSETS
        log.info(
            "Loading OS-Atlas subsets %s [first %d rows]", subsets, args.train_subset_size
        )
        return os_atlas_dataset(subsets, limit=args.train_subset_size)

    from datasets import load_dataset

    log.info(
        "Streaming %s [first %d rows]", args.train_dataset, args.train_subset_size
    )
    ds = load_dataset(args.train_dataset, split="train", streaming=True)
    return ds.take(args.train_subset_size)


def _is_qwen_bad_vision_batch_error(exc: BaseException) -> bool:
    """True for Qwen2.5-VL's known bad-image spatial-merge crash."""
    msg = str(exc)
    return (
        "shape '[0, 4, -1]' is invalid" in msg
        or 'shape "[0, 4, -1]" is invalid' in msg
    )


def _zero_trainable_loss(model: Any) -> Any:
    for param in model.parameters():
        if getattr(param, "requires_grad", False):
            return param.sum() * 0.0
    for param in model.parameters():
        return param.sum() * 0.0
    raise RuntimeError("Cannot build zero loss for a model with no parameters")


def _skip_bad_vision_trainer(base_cls: type) -> type:
    """Return a Trainer subclass that skips Qwen malformed visual batches.

    Some UGround rows still produce a Qwen vision-tower reshape crash after
    processor-level validation, depending on the Colab transformers/torch path.
    Treat those rare rows like corrupt training examples instead of aborting
    the whole LoRA run.
    """

    class SkipBadVisionTrainer(base_cls):
        _bad_vision_batches: int = 0

        def compute_loss(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            try:
                return super().compute_loss(model, inputs, *args, **kwargs)
            except RuntimeError as exc:
                if not _is_qwen_bad_vision_batch_error(exc):
                    raise
                self._bad_vision_batches += 1
                if self._bad_vision_batches <= 5 or self._bad_vision_batches % 25 == 0:
                    log.warning(
                        "Skipping Qwen bad visual batch #%d: %s",
                        self._bad_vision_batches,
                        exc,
                    )
                loss = _zero_trainable_loss(model)
                if kwargs.get("return_outputs"):
                    return loss, {}
                return loss

    return SkipBadVisionTrainer
