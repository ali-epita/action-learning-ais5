"""Grounding-data adapters and a VLM-aware training collator.

Task 1's LoRA loop needs (image, instruction, target point) turned into
tokenized model inputs with masked labels. This module:

  - Defines a small `GroundingTrainExample` shape every adapter targets.
  - Provides format adapters for the dataset schemas the proposal mentions
    (OS-Atlas, UGround, a heuristic fallback for anything else).
  - Provides `QwenVLGroundingCollator`, an HF-Trainer compatible callable that
    turns a list of examples into a batch of {input_ids, attention_mask,
    pixel_values, image_grid_thw, labels} where every label position before
    the assistant answer is set to -100.

The collator builds the same prompt as `ais5.models.qwen.QwenVL.predict` so
train- and test-time prompts match. The assistant target string matches the
`qwen-click` parser regex in `ais5.prompt.action`, so an adapter trained with
this collator emits responses the inference parser can read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image as PILImage
from PIL.Image import Image

from ..prompt.templates import CLICK_PROMPT, format_click_prompt


@dataclass
class GroundingTrainExample:
    """Internal training shape: PIL image, instruction, target click point."""

    image: Image
    instruction: str
    target_point: tuple[float, float]  # (x, y) in image pixels
    sample_id: str | None = None


def _coerce_image(obj: Any) -> Image | None:
    """Turn a HF-datasets `image` field into a PIL RGB image, or None."""
    if obj is None:
        return None
    if isinstance(obj, Image):
        return obj if obj.mode == "RGB" else obj.convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return PILImage.open(BytesIO(obj)).convert("RGB")
    if isinstance(obj, str):
        return PILImage.open(obj).convert("RGB")
    if isinstance(obj, dict):
        # HF datasets can hand back {"bytes": b"...", "path": "..."}.
        if obj.get("bytes") is not None:
            return PILImage.open(BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path"):
            return PILImage.open(obj["path"]).convert("RGB")
    return None


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    """Return the centroid of an (x1, y1, x2, y2) bbox, or None if malformed."""
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def adapt_os_atlas_row(row: dict) -> GroundingTrainExample | None:
    """Adapter for `OS-Copilot/OS-Atlas-data` style rows.

    Expected fields: `image` (PIL or bytes-dict), `instruction` (str), and
    either `bbox` (x1, y1, x2, y2) or `point` (x, y). Click target is the
    bbox center when only a bbox is available.
    """
    image = _coerce_image(row.get("image"))
    instruction = row.get("instruction") or row.get("query")
    point = _coerce_point(row.get("point") or row.get("target_point"))
    if point is None:
        point = _bbox_center(row.get("bbox") or row.get("target_bbox"))
    if image is None or not instruction or point is None:
        return None
    return GroundingTrainExample(
        image=image,
        instruction=str(instruction),
        target_point=point,
        sample_id=row.get("id") or row.get("sample_id"),
    )


def adapt_uground_row(row: dict) -> GroundingTrainExample | None:
    """Adapter for UGround rows. In practice the same shape as OS-Atlas."""
    return adapt_os_atlas_row(row)


def adapt_auto_row(row: dict) -> GroundingTrainExample | None:
    """Heuristic adapter that tries common field names for unknown sources."""
    image = _coerce_image(row.get("image") or row.get("screenshot") or row.get("img"))
    instruction = (
        row.get("instruction")
        or row.get("query")
        or row.get("task")
        or row.get("prompt")
        or row.get("text")
    )
    point = _coerce_point(row.get("point") or row.get("target_point") or row.get("click"))
    if point is None:
        point = _bbox_center(row.get("bbox") or row.get("target_bbox"))
    if image is None or not instruction or point is None:
        return None
    return GroundingTrainExample(
        image=image,
        instruction=str(instruction),
        target_point=point,
        sample_id=row.get("id") or row.get("sample_id"),
    )


def _coerce_point(obj: Any) -> tuple[float, float] | None:
    if obj is None:
        return None
    try:
        x, y = obj
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


ROW_ADAPTERS: dict[str, Callable[[dict], GroundingTrainExample | None]] = {
    "os-atlas": adapt_os_atlas_row,
    "uground": adapt_uground_row,
    "auto": adapt_auto_row,
}


def stream_grounding_examples(
    rows: Iterable[dict],
    *,
    adapter: str | Callable[[dict], GroundingTrainExample | None] = "auto",
    limit: int | None = None,
) -> Iterable[GroundingTrainExample]:
    """Yield `GroundingTrainExample`s from raw dataset rows.

    `adapter` is either a key in `ROW_ADAPTERS` or a callable. Rows that fail
    to adapt are skipped, so a partial dataset still trains; the caller can
    count yielded examples to log the survival rate.
    """
    if isinstance(adapter, str):
        if adapter not in ROW_ADAPTERS:
            raise ValueError(
                f"Unknown adapter {adapter!r}. Available: {sorted(ROW_ADAPTERS)}"
            )
        fn = ROW_ADAPTERS[adapter]
    else:
        fn = adapter

    n = 0
    for row in rows:
        if limit is not None and n >= limit:
            return
        ex = fn(row)
        if ex is None:
            continue
        yield ex
        n += 1


def _format_answer(point: tuple[float, float]) -> str:
    """Assistant target string. Matches the `qwen-click` parser regex."""
    x, y = point
    return f"<click>{int(round(x))}, {int(round(y))}</click>"


class QwenVLGroundingCollator:
    """HF-Trainer compatible collator for Qwen2.5-VL grounding fine-tuning.

    For each example, builds:

        user:      {image} + format_click_prompt(instruction)
        assistant: <click>x, y</click>

    Calls the processor twice per example: once with the prompt only (to
    locate the prefix length), once for the full conversation. Labels =
    input_ids with the prefix and padding masked to `ignore_index`.

    The two-pass tokenization is correct because the image expansion is
    deterministic for a fixed image, so the prompt-only token count equals
    the prefix length of the full sequence (right padding assumed, which is
    the default for HF training).
    """

    def __init__(
        self,
        processor: Any,
        *,
        prompt_template: str | None = None,
        ignore_index: int = -100,
    ) -> None:
        self.processor = processor
        self.prompt_template = prompt_template or CLICK_PROMPT
        self.ignore_index = ignore_index
        # Qwen's tokenizer often ships without a pad token; reuse EOS so HF
        # Trainer can right-pad batches.
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token", None) is None:
            eos = getattr(tok, "eos_token", None)
            if eos is not None:
                tok.pad_token = eos

    def _build_messages(
        self, example: GroundingTrainExample, *, with_answer: bool
    ) -> list[dict[str, Any]]:
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": example.image},
                {
                    "type": "text",
                    "text": format_click_prompt(
                        example.instruction, template=self.prompt_template
                    ),
                },
            ],
        }
        if not with_answer:
            return [user]
        return [
            user,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": _format_answer(example.target_point)}
                ],
            },
        ]

    def __call__(self, examples: list[GroundingTrainExample]) -> dict[str, Any]:
        # Pass 1: per-example prompt-only length (where the answer starts).
        prompt_lens: list[int] = []
        for ex in examples:
            msgs = self._build_messages(ex, with_answer=False)
            prompt_text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.processor(
                text=[prompt_text],
                images=[ex.image],
                return_tensors="pt",
                padding=False,
            )
            prompt_lens.append(int(prompt_tokens["input_ids"].shape[1]))

        # Pass 2: batch the full conversations.
        full_texts: list[str] = []
        images: list[Image] = []
        for ex in examples:
            msgs = self._build_messages(ex, with_answer=True)
            full_texts.append(
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(ex.image)
        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            # Defensive clamp in case truncation made the prefix shorter than
            # measured (no truncation today, but cheap to guard).
            labels[i, : min(plen, labels.shape[1])] = self.ignore_index

        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = self.ignore_index
        else:
            tok = getattr(self.processor, "tokenizer", None)
            pad_id = getattr(tok, "pad_token_id", None) if tok is not None else None
            if pad_id is not None:
                labels[input_ids == pad_id] = self.ignore_index

        out = dict(batch)
        out["labels"] = labels
        return out


def make_collator(
    backbone: str,
    processor: Any,
    *,
    adapter: str | Callable[[dict], GroundingTrainExample | None] | None = None,
    prompt_template: str | None = None,
) -> Callable[[list[Any]], dict[str, Any]]:
    """Return the right collator for `backbone`.

    Today only Qwen2.5-VL is wired up. PaliGemma needs a separate collator
    because the prompt format and image-token expansion differ.

    If `adapter` is given, the returned callable accepts raw HF-dataset rows
    (dicts) and converts each via the adapter before batching. If `adapter`
    is None, the callable expects pre-built `GroundingTrainExample`s.
    """
    family = backbone.lower()
    if "qwen" in family:
        base = QwenVLGroundingCollator(processor, prompt_template=prompt_template)
    else:
        raise NotImplementedError(
            f"No collator yet for backbone {backbone!r}. Today: 'qwen2.5-vl'."
        )

    if adapter is None:
        return base

    if isinstance(adapter, str):
        if adapter not in ROW_ADAPTERS:
            raise ValueError(
                f"Unknown adapter {adapter!r}. Available: {sorted(ROW_ADAPTERS)}"
            )
        adapter_fn = ROW_ADAPTERS[adapter]
    else:
        adapter_fn = adapter

    def _row_collator(rows: list[Any]) -> dict[str, Any]:
        examples: list[GroundingTrainExample] = []
        for r in rows:
            if isinstance(r, GroundingTrainExample):
                examples.append(r)
                continue
            ex = adapter_fn(r)
            if ex is not None:
                examples.append(ex)
        if not examples:
            raise ValueError(
                "Collator received a batch where every row failed to adapt — "
                "check that the dataset schema matches the adapter."
            )
        return base(examples)

    return _row_collator
