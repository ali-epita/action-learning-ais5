"""TrainingArgs carries the disk-bounding defaults the Colab plan relies on."""

from __future__ import annotations

from ais5.adapt.train import TrainingArgs


def test_checkpoint_disk_defaults():
    a = TrainingArgs()
    assert a.save_total_limit == 1
    assert a.gradient_checkpointing is False


def test_extra_overrides_explicit_fields():
    # extra is spread last in run_lora_training, so it wins over the field value.
    a = TrainingArgs(save_total_limit=1, extra={"save_total_limit": 3})
    merged = {"save_total_limit": a.save_total_limit, **a.extra}
    assert merged["save_total_limit"] == 3
