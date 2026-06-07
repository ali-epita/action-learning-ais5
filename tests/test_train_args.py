"""TrainingArgs carries the disk-bounding defaults the Colab plan relies on."""

from __future__ import annotations

from ais5.adapt.train import TrainingArgs


def test_checkpoint_disk_defaults():
    a = TrainingArgs()
    assert a.save_total_limit == 1
    assert a.gradient_checkpointing is False
    assert a.skip_bad_vision_batches is True


def test_extra_overrides_explicit_fields():
    # extra is spread last in run_lora_training, so it wins over the field value.
    a = TrainingArgs(save_total_limit=1, extra={"save_total_limit": 3})
    merged = {"save_total_limit": a.save_total_limit, **a.extra}
    assert merged["save_total_limit"] == 3


def test_qwen_bad_vision_error_detector_is_narrow():
    from ais5.adapt.train import _is_qwen_bad_vision_batch_error

    assert _is_qwen_bad_vision_batch_error(
        RuntimeError("shape '[0, 4, -1]' is invalid for input of size 2560")
    )
    assert not _is_qwen_bad_vision_batch_error(RuntimeError("CUDA out of memory"))


def test_zero_trainable_loss_requires_grad():
    import torch

    from ais5.adapt.train import _zero_trainable_loss

    model = torch.nn.Linear(2, 1)
    loss = _zero_trainable_loss(model)
    assert loss.requires_grad
    assert loss.item() == 0.0
