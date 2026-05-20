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

    with pytest.raises(NotImplementedError) as exc_info:
        make_collator("llava", _FakeProcessor())
    # Error lists supported backbones so the user knows what to switch to.
    msg = str(exc_info.value)
    assert "qwen" in msg.lower() and "paligemma" in msg.lower()


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


# ── PaliGemma collator ───────────────────────────────────────────────────────


class _FakePaliGemmaProcessor:
    """Minimal PaliGemmaProcessor stand-in.

    When `suffix=` is supplied, returns `labels` with the prompt prefix masked,
    mirroring the real processor's behavior. With `return_labels=False` it omits
    `labels` entirely, simulating a too-old transformers version.
    """

    def __init__(self, *, return_labels: bool = True) -> None:
        self.tokenizer = _FakeTokenizer()
        self.return_labels = return_labels

    def __call__(
        self,
        *,
        text: list[str],
        images: list,
        suffix: list[str] | None = None,
        return_tensors: str = "pt",
        padding: str = "longest",
    ) -> dict:
        assert return_tensors == "pt"
        prefix_lens: list[int] = []
        full_seqs: list[list[int]] = []
        for i, prompt in enumerate(text):
            prefix = [99, 99, 99] + [(ord(c) % 90) + 10 for c in prompt]
            prefix_lens.append(len(prefix))
            if suffix is not None:
                suf_ids = [(ord(c) % 90) + 10 for c in suffix[i]]
                full_seqs.append(prefix + suf_ids)
            else:
                full_seqs.append(list(prefix))
        max_len = max(len(s) for s in full_seqs)
        input_ids = torch.full(
            (len(full_seqs), max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attn = torch.zeros_like(input_ids)
        for i, s in enumerate(full_seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        result: dict = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pixel_values": torch.ones(len(full_seqs), 3, 8, 8),
        }
        if suffix is not None and self.return_labels:
            labels = input_ids.clone()
            for i, pl in enumerate(prefix_lens):
                labels[i, :pl] = -100
            labels[attn == 0] = -100
            result["labels"] = labels
        return result


def test_paligemma_encode_format_matches_parser_regex():
    import re

    from ais5.adapt.data import _encode_paligemma_loc
    from ais5.prompt.action import parse_click

    s = _encode_paligemma_loc((128, 256), image_size=(1024, 512), bins=1024)
    assert re.fullmatch(r"<loc\d{4}><loc\d{4}>", s) is not None

    # Round-trip through the decoder. The parser returns bin centers, so the
    # decoded point is within half a bin of the original target.
    parsed = parse_click(s, image_size=(1024, 512))
    assert parsed.point is not None
    px, py = parsed.point
    assert abs(px - 128.5) < 1.0
    assert abs(py - 256.25) < 1.0


def test_paligemma_encode_clamps_to_grid_edge():
    from ais5.adapt.data import _encode_paligemma_loc

    # Pixel coords sitting on the image edge would index past the grid without
    # the clamp; the collator must never emit <loc1024>.
    assert _encode_paligemma_loc((100, 100), image_size=(100, 100), bins=1024) == \
        "<loc1023><loc1023>"


def test_paligemma_collator_returns_labels_with_prefix_masked():
    from ais5.adapt import PaliGemmaGroundingCollator

    proc = _FakePaliGemmaProcessor()
    collator = PaliGemmaGroundingCollator(proc)
    ex = _make_example(target=(50, 50), instruction="click")
    batch = collator([ex])

    assert "labels" in batch
    labels = batch["labels"]
    # First three positions are the image tokens — must be masked.
    assert (labels[0, :3] == -100).all()
    # At least some positions are real (the suffix tokens).
    assert (labels[0] != -100).any()


def test_paligemma_collator_raises_when_processor_no_labels():
    """If the processor is too old to emit suffix-derived labels, we fail loudly."""
    from ais5.adapt import PaliGemmaGroundingCollator

    proc = _FakePaliGemmaProcessor(return_labels=False)
    collator = PaliGemmaGroundingCollator(proc)
    with pytest.raises(RuntimeError, match="did not return"):
        collator([_make_example()])


def test_paligemma_collator_batches_two_examples():
    from ais5.adapt import PaliGemmaGroundingCollator

    proc = _FakePaliGemmaProcessor()
    collator = PaliGemmaGroundingCollator(proc)
    examples = [
        _make_example(target=(10, 20), instruction="short"),
        _make_example(target=(40, 60), instruction="much longer instruction"),
    ]
    batch = collator(examples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    # Shorter sequence's padded positions are masked.
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_make_collator_paligemma_dispatch():
    from ais5.adapt import PaliGemmaGroundingCollator, make_collator

    proc = _FakePaliGemmaProcessor()
    collator = make_collator("paligemma", proc)
    assert isinstance(collator, PaliGemmaGroundingCollator)


def test_make_collator_paligemma_with_adapter():
    from ais5.adapt import make_collator

    proc = _FakePaliGemmaProcessor()
    collator = make_collator("paligemma", proc, adapter="auto")
    rows = [
        {"image": _img(), "instruction": "tap", "point": (5.0, 5.0)},
    ]
    batch = collator(rows)
    assert "labels" in batch


