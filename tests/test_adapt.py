"""CPU-only tests for Task 1's adapt module.

Covers: LoRAConfig.to_peft round-trip, attach_lora + count_trainable on a
toy module, row adapters and the streaming generator, and the QwenVLGroundingCollator
batching / label-masking behavior against a fake processor.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
PILImage = pytest.importorskip("PIL.Image")
peft = pytest.importorskip("peft")


# ── Fake Qwen-style processor used by the collator tests ─────────────────────


class _FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1


class _FakeProcessor:
    """Minimal Qwen2.5-VL processor stand-in.

    Tokenization is deterministic: 3 fake image-pad tokens per image followed
    by one token per character of the chat-templated text. That lets the tests
    reason precisely about where the prompt ends and the answer begins.
    """

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False, "test fake only supports tokenize=False"
        out: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content_str = "".join(texts)
            else:
                content_str = str(content)
            out.append(f"|{role}|{content_str}|")
        if add_generation_prompt:
            out.append("|assistant|")
        return "".join(out)

    def __call__(
        self,
        *,
        text: list[str],
        images: list,
        return_tensors: str = "pt",
        padding: bool = False,
    ) -> dict:
        assert return_tensors == "pt"
        per_seq: list[list[int]] = []
        for t, _img in zip(text, images, strict=False):
            ids = [99, 99, 99] + [(ord(c) % 90) + 10 for c in t]
            per_seq.append(ids)
        max_len = max(len(s) for s in per_seq)
        out_ids = torch.full(
            (len(per_seq), max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        out_attn = torch.zeros_like(out_ids)
        for i, ids in enumerate(per_seq):
            out_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            out_attn[i, : len(ids)] = 1
        return {
            "input_ids": out_ids,
            "attention_mask": out_attn,
            "pixel_values": torch.ones(len(per_seq), 3, 8, 8),
            "image_grid_thw": torch.ones(len(per_seq), 1, dtype=torch.long),
        }


def _img(w: int = 64, h: int = 64) -> "PILImage.Image":
    return PILImage.new("RGB", (w, h), color="white")


def _make_example(target=(100, 200), instruction="click here"):
    from ais5.adapt import GroundingTrainExample

    return GroundingTrainExample(image=_img(), instruction=instruction, target_point=target)


# ── LoRAConfig + attach_lora + count_trainable ───────────────────────────────


def test_lora_config_to_peft_roundtrip():
    from ais5.adapt import LoRAConfig

    cfg = LoRAConfig(r=8, alpha=16, backbone="qwen2.5-vl")
    pcfg = cfg.to_peft()
    assert pcfg.r == 8
    assert pcfg.lora_alpha == 16
    assert set(pcfg.target_modules) == set(cfg.resolved_targets())


def test_lora_config_default_targets_per_backbone():
    from ais5.adapt import LoRAConfig

    qwen_targets = LoRAConfig(backbone="qwen2.5-vl").resolved_targets()
    pali_targets = LoRAConfig(backbone="paligemma").resolved_targets()
    assert "q_proj" in qwen_targets
    assert "q_proj" in pali_targets


def test_attach_lora_and_count_trainable_on_toy_module():
    import torch.nn as nn

    from ais5.adapt import LoRAConfig, attach_lora, count_trainable

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.other = nn.Linear(8, 4)

        def forward(self, x):
            return self.other(self.q_proj(x) + self.v_proj(x))

    base = Toy()
    # `FEATURE_EXTRACTION` avoids PEFT's CausalLM-only assumption that the
    # base module implements `prepare_inputs_for_generation`. Real backbones
    # (Qwen2.5-VL, PaliGemma) keep the CAUSAL_LM default in LoRAConfig.
    cfg = LoRAConfig(
        r=4,
        alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = attach_lora(base, cfg)
    trainable, total = count_trainable(peft_model)
    assert 0 < trainable < total, "Some params should be trainable (LoRA), not all"


# ── Row adapters ─────────────────────────────────────────────────────────────


def test_adapt_os_atlas_row_with_bbox():
    from ais5.adapt import adapt_os_atlas_row

    row = {"image": _img(), "instruction": "click", "bbox": [10, 20, 30, 40]}
    ex = adapt_os_atlas_row(row)
    assert ex is not None
    assert ex.target_point == (20.0, 30.0)  # bbox center
    assert ex.instruction == "click"


def test_adapt_os_atlas_row_with_explicit_point():
    from ais5.adapt import adapt_os_atlas_row

    row = {"image": _img(), "instruction": "go", "point": (3.5, 7.5)}
    ex = adapt_os_atlas_row(row)
    assert ex is not None
    assert ex.target_point == (3.5, 7.5)


def test_adapt_os_atlas_row_returns_none_on_missing_fields():
    from ais5.adapt import adapt_os_atlas_row

    # Missing instruction
    assert adapt_os_atlas_row({"image": _img(), "bbox": [1, 2, 3, 4]}) is None
    # Missing image
    assert adapt_os_atlas_row({"instruction": "x", "bbox": [1, 2, 3, 4]}) is None
    # Missing point and bbox
    assert adapt_os_atlas_row({"image": _img(), "instruction": "x"}) is None


def test_adapt_auto_row_with_alternative_field_names():
    from ais5.adapt import adapt_auto_row

    row = {"screenshot": _img(), "task": "tap", "point": (12.0, 8.0)}
    ex = adapt_auto_row(row)
    assert ex is not None
    assert ex.target_point == (12.0, 8.0)


def test_stream_grounding_examples_skips_invalid():
    from ais5.adapt import stream_grounding_examples

    rows = [
        {"image": _img(), "instruction": "a", "bbox": [1, 2, 3, 4]},
        {"image": None},  # gets skipped
        {"image": _img(), "instruction": "b", "point": (5.0, 5.0)},
    ]
    examples = list(stream_grounding_examples(rows, adapter="auto"))
    assert len(examples) == 2
    assert examples[0].instruction == "a"
    assert examples[1].instruction == "b"


def test_stream_grounding_examples_honors_limit():
    from ais5.adapt import stream_grounding_examples

    rows = [
        {"image": _img(), "instruction": f"i{i}", "point": (1.0, 1.0)} for i in range(10)
    ]
    examples = list(stream_grounding_examples(rows, adapter="auto", limit=3))
    assert len(examples) == 3


def test_stream_grounding_examples_unknown_adapter_raises():
    from ais5.adapt import stream_grounding_examples

    with pytest.raises(ValueError, match="Unknown adapter"):
        list(stream_grounding_examples([], adapter="nonexistent"))


# ── Collator ─────────────────────────────────────────────────────────────────


def test_qwen_collator_batch_shape_and_keys():
    from ais5.adapt import QwenVLGroundingCollator

    proc = _FakeProcessor()
    collator = QwenVLGroundingCollator(proc)
    examples = [
        _make_example(target=(50, 75), instruction="A"),
        _make_example(target=(100, 200), instruction="BBBB"),
    ]
    batch = collator(examples)

    assert {"input_ids", "attention_mask", "pixel_values", "labels"}.issubset(batch)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == 2


def test_qwen_collator_masks_prompt_prefix():
    from ais5.adapt import QwenVLGroundingCollator

    proc = _FakeProcessor()
    collator = QwenVLGroundingCollator(proc)
    ex = _make_example(target=(50, 75), instruction="A")
    batch = collator([ex])

    # Re-derive the prompt-only length the same way the collator does internally.
    prompt_msgs = collator._build_messages(ex, with_answer=False)
    prompt_text = proc.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_only = proc(
        text=[prompt_text], images=[ex.image], return_tensors="pt", padding=False
    )
    prompt_len = int(prompt_only["input_ids"].shape[1])

    labels = batch["labels"]
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]

    # Everything in the prompt prefix is ignored.
    assert (labels[0, :prompt_len] == -100).all(), (
        f"prefix not masked; got {labels[0, :prompt_len].tolist()}"
    )

    # Everything in the answer span (after the prefix, where attention is on)
    # matches the input ids verbatim.
    answer_span = slice(prompt_len, attn.shape[1])
    on = attn[0, answer_span] == 1
    assert (labels[0, answer_span][on] == input_ids[0, answer_span][on]).all()


def test_qwen_collator_masks_padding():
    from ais5.adapt import QwenVLGroundingCollator

    proc = _FakeProcessor()
    collator = QwenVLGroundingCollator(proc)
    # Different instruction lengths force the shorter sequence to be padded.
    short = _make_example(target=(1, 1), instruction="A")
    longer = _make_example(target=(2, 2), instruction="A" * 30)
    batch = collator([short, longer])

    labels = batch["labels"]
    attention = batch["attention_mask"]
    # Every padded position must be -100.
    assert (labels[attention == 0] == -100).all()


def test_qwen_collator_sets_pad_token_when_missing():
    """If the tokenizer ships without a pad_token, the collator reuses EOS."""
    from ais5.adapt import QwenVLGroundingCollator

    proc = _FakeProcessor()
    proc.tokenizer.pad_token = None  # simulate Qwen's "no pad" default
    QwenVLGroundingCollator(proc)
    assert proc.tokenizer.pad_token == proc.tokenizer.eos_token


# ── make_collator ────────────────────────────────────────────────────────────


def test_make_collator_with_adapter_string_accepts_raw_rows():
    from ais5.adapt import make_collator

    proc = _FakeProcessor()
    collator = make_collator("qwen2.5-vl", proc, adapter="auto")
    rows = [
        {"image": _img(), "instruction": "tap", "point": (5.0, 5.0)},
        {"image": _img(), "instruction": "scroll", "point": (10.0, 20.0)},
    ]
    batch = collator(rows)
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 2


def test_make_collator_without_adapter_expects_examples():
    from ais5.adapt import make_collator

    proc = _FakeProcessor()
    collator = make_collator("qwen2.5-vl", proc)
    batch = collator([_make_example()])
    assert "labels" in batch


def test_make_collator_rejects_unknown_backbone():
    from ais5.adapt import make_collator

    with pytest.raises(NotImplementedError):
        make_collator("paligemma", _FakeProcessor())


def test_make_collator_rejects_unknown_adapter():
    from ais5.adapt import make_collator

    with pytest.raises(ValueError, match="Unknown adapter"):
        make_collator("qwen2.5-vl", _FakeProcessor(), adapter="bogus")


def test_make_collator_with_adapter_raises_if_all_rows_fail():
    from ais5.adapt import make_collator

    proc = _FakeProcessor()
    collator = make_collator("qwen2.5-vl", proc, adapter="auto")
    # Rows with no usable image or target.
    with pytest.raises(ValueError, match="failed to adapt"):
        collator([{"foo": 1}, {"bar": 2}])
