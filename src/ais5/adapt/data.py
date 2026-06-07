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

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image as PILImage
from PIL.Image import Image

from ..models.paligemma import PALIGEMMA_POINT_PROMPT
from ..prompt.templates import CLICK_PROMPT, format_click_prompt


@dataclass
class GroundingTrainExample:
    """Internal training shape: PIL image, instruction, target click point."""

    image: Image
    instruction: str
    target_point: tuple[float, float]  # (x, y) in image pixels
    sample_id: str | None = None


def _coerce_image(obj: Any) -> Image | None:
    """Turn a HF-datasets `image` field into a PIL RGB image, or None.

    Forces decode via `.load()` so corrupt or truncated bytes raise here
    (and we skip the row) rather than producing a PIL handle that looks
    valid by `.width`/`.height` but holds degenerate pixel data, which
    then breaks downstream vision-tower reshapes.
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, Image):
            img = obj if obj.mode == "RGB" else obj.convert("RGB")
        elif isinstance(obj, (bytes, bytearray)):
            img = PILImage.open(BytesIO(obj)).convert("RGB")
        elif isinstance(obj, str):
            img = PILImage.open(obj).convert("RGB")
        elif isinstance(obj, dict):
            # HF datasets can hand back {"bytes": b"...", "path": "..."}.
            if obj.get("bytes") is not None:
                img = PILImage.open(BytesIO(obj["bytes"])).convert("RGB")
            elif obj.get("path"):
                img = PILImage.open(obj["path"]).convert("RGB")
            else:
                return None
        else:
            return None
        img.load()
    except Exception:  # noqa: BLE001
        return None
    return img


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


_UGROUND_DESC_RE = re.compile(r"Description:\s*(.+?)\s*Answer:", re.DOTALL)
_UGROUND_POINT_RE = re.compile(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)")


def adapt_uground_row(row: dict) -> GroundingTrainExample | None:
    """Adapter for `osunlp/UGround-V1-Data` (LLaVA-style conversations).

    Each row carries `{image (bytes), conversations (list[dict] or JSON string),
    width, height}`. The human turn embeds the target description between
    "Description:" and "Answer:"; the gpt turn answers with "(x, y)" in pixel
    coordinates relative to the image.

    Tiny images (<56px on either side) are skipped — Qwen2.5-VL's vision tower
    requires at least 4 patches (2x2 merge unit) after preprocessing, and
    sub-thumbnail screenshots break the spatial_merge reshape.
    """
    image = _coerce_image(row.get("image"))
    if image is None or image.width < 56 or image.height < 56:
        return None
    convs = row.get("conversations")
    if isinstance(convs, str):
        try:
            convs = json.loads(convs)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(convs, list):
        return None
    human = next((t.get("value") for t in convs if t.get("from") == "human"), None)
    gpt = next((t.get("value") for t in convs if t.get("from") == "gpt"), None)
    if not human or not gpt:
        return None
    desc_match = _UGROUND_DESC_RE.search(human)
    instruction = desc_match.group(1).strip() if desc_match else human.strip()
    point_match = _UGROUND_POINT_RE.search(gpt)
    if point_match is None:
        return None
    point = (float(point_match.group(1)), float(point_match.group(2)))
    return GroundingTrainExample(
        image=image,
        instruction=instruction,
        target_point=point,
    )


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

    # Qwen2.5-VL's vision tower applies a 2x2 spatial merge (spatial_merge_unit=4),
    # so images with fewer than 4 raw patches, or a non-multiple-of-4 patch
    # count, crash the reshape. We pre-validate per-example in pass 1 and drop
    # offenders before they reach the model.
    _MIN_VISION_TOKENS = 4
    _SPATIAL_MERGE_UNIT = 4
    _RAW_PATCH_NUMEL = 3 * 2 * 14 * 14

    @classmethod
    def _valid_vision_shape(cls, tokens: dict[str, Any]) -> bool:
        grid = tokens.get("image_grid_thw")
        grid_patches: list[int] = []
        if grid is not None and grid.numel() > 0:
            for row in grid:
                n_visual = int(row.prod().item())
                grid_patches.append(n_visual)
                if n_visual < cls._MIN_VISION_TOKENS:
                    return False
                if n_visual % cls._SPATIAL_MERGE_UNIT != 0:
                    return False

        # Real Qwen2.5-VL processors can return either flattened patch rows
        # `(n_patches, 1176)` or explicit raw patches `(n_patches, 3, 2, 14, 14)`.
        # The vision tower reshapes this tensor before the 2x2 merge, so validate
        # the effective patch-row count and its consistency with image_grid_thw.
        pixel_values = tokens.get("pixel_values")
        n_patches: int | None = None
        if pixel_values is not None:
            ndim = getattr(pixel_values, "ndim", None)
            shape = tuple(getattr(pixel_values, "shape", ()))
            if ndim == 2:
                n_patches = int(shape[0])
            elif ndim == 5 and shape[1:] == (3, 2, 14, 14):
                n_patches = int(shape[0])
            elif hasattr(pixel_values, "numel") and pixel_values.numel() % cls._RAW_PATCH_NUMEL == 0:
                n_patches = int(pixel_values.numel() // cls._RAW_PATCH_NUMEL)

        if n_patches is not None:
            if n_patches < cls._MIN_VISION_TOKENS:
                return False
            if n_patches % cls._SPATIAL_MERGE_UNIT != 0:
                return False
            if grid_patches and n_patches != sum(grid_patches):
                return False

        return True

    def __call__(self, examples: list[GroundingTrainExample]) -> dict[str, Any]:
        # Pass 1: per-example prompt-only length + validate visual-token count.
        valid_examples: list[GroundingTrainExample] = []
        prompt_lens: list[int] = []
        for ex in examples:
            try:
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
            except Exception:  # noqa: BLE001
                continue
            if not self._valid_vision_shape(prompt_tokens):
                continue

            try:
                full_msgs = self._build_messages(ex, with_answer=True)
                full_text = self.processor.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
                full_tokens = self.processor(
                    text=[full_text],
                    images=[ex.image],
                    return_tensors="pt",
                    padding=False,
                )
            except Exception:  # noqa: BLE001
                continue
            if not self._valid_vision_shape(full_tokens):
                continue

            valid_examples.append(ex)
            prompt_lens.append(int(prompt_tokens["input_ids"].shape[1]))

        if not valid_examples:
            raise ValueError(
                "QwenVLGroundingCollator: every example in this batch produced "
                "too few visual tokens for the 2x2 spatial merger. Increase "
                "per_device_train_batch_size or tighten the adapter filter."
            )

        # Pass 2: batch the full conversations for the survivors only.
        full_texts: list[str] = []
        images: list[Image] = []
        for ex in valid_examples:
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
        if not self._valid_vision_shape(batch):
            raise ValueError(
                "QwenVLGroundingCollator: the batched full conversation still "
                "has an invalid visual-token shape after per-example filtering."
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


def _encode_paligemma_loc(point: tuple[float, float], image_size: tuple[int, int], bins: int) -> str:
    """Bin a pixel target into the `<loc{row:04d}><loc{col:04d}>` PaliGemma format.

    Uses floor binning so coordinates in `[0, span)` land in their natural bin;
    clamps to `bins - 1` so pixel coords sitting on the bottom/right edge
    don't index past the grid. The decoder in `ais5.prompt.action` reads bin
    centers, so floor encoding is the correct inverse for the median case.
    """
    x, y = point
    w, h = image_size
    col = min(max(int(x / max(1, w) * bins), 0), bins - 1)
    row = min(max(int(y / max(1, h) * bins), 0), bins - 1)
    return f"<loc{row:04d}><loc{col:04d}>"


class PaliGemmaGroundingCollator:
    """HF-Trainer compatible collator for PaliGemma grounding fine-tuning.

    PaliGemma was pre-trained with task prefixes; for pointing tasks the prompt
    is `"point: <instruction>"` and the target is `<loc{row:04d}><loc{col:04d}>`,
    indexing a `bins`-wide grid that the `paligemma-loc` parser decodes back
    to pixels.

    Uses PaliGemmaProcessor's `suffix=` argument so the prefix portion of
    `labels` is auto-masked. Requires transformers>=4.40 (the version Colab
    ships with `transformers<5` is well past that).
    """

    def __init__(
        self,
        processor: Any,
        *,
        prompt_template: str | None = None,
        bins: int = 1024,
        ignore_index: int = -100,
    ) -> None:
        self.processor = processor
        self.prompt_template = prompt_template or PALIGEMMA_POINT_PROMPT
        self.bins = bins
        self.ignore_index = ignore_index
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token", None) is None:
            eos = getattr(tok, "eos_token", None)
            if eos is not None:
                tok.pad_token = eos

    def _build_prompt(self, example: GroundingTrainExample) -> str:
        return self.prompt_template.format(instruction=example.instruction)

    def _format_answer(self, example: GroundingTrainExample) -> str:
        return _encode_paligemma_loc(
            example.target_point, example.image.size, self.bins
        )

    def __call__(self, examples: list[GroundingTrainExample]) -> dict[str, Any]:
        prompts = [self._build_prompt(ex) for ex in examples]
        suffixes = [self._format_answer(ex) for ex in examples]
        images = [ex.image for ex in examples]
        batch = self.processor(
            text=prompts,
            suffix=suffixes,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        if "labels" not in batch:
            raise RuntimeError(
                "PaliGemmaProcessor did not return `labels` for the suffix= "
                "path. Upgrade transformers (>=4.40) or pass a custom collator."
            )
        return dict(batch)


def make_collator(
    backbone: str,
    processor: Any,
    *,
    adapter: str | Callable[[dict], GroundingTrainExample | None] | None = None,
    prompt_template: str | None = None,
) -> Callable[[list[Any]], dict[str, Any]]:
    """Return the right collator for `backbone`.

    Today: 'qwen2.5-vl' and 'paligemma'. Other backbones raise
    NotImplementedError. The two implementations differ because the prompt
    format and target-token convention differ:

      qwen2.5-vl : chat template, target = "<click>x, y</click>"
      paligemma  : "point: ..." prefix, target = "<loc{row:04d}><loc{col:04d}>"

    If `adapter` is given, the returned callable accepts raw HF-dataset rows
    (dicts) and converts each via the adapter before batching. If `adapter`
    is None, the callable expects pre-built `GroundingTrainExample`s.
    """
    family = backbone.lower()
    if "qwen" in family:
        base = QwenVLGroundingCollator(processor, prompt_template=prompt_template)
    elif "paligemma" in family:
        base = PaliGemmaGroundingCollator(processor, prompt_template=prompt_template)
    else:
        raise NotImplementedError(
            f"No collator yet for backbone {backbone!r}. Today: 'qwen2.5-vl', 'paligemma'."
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
