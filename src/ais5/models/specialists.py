"""GUI-specialist wrappers: OS-Atlas, ShowUI, Ferret-UI Lite.

These are baselines (no fine-tuning by us). All three were trained as
Qwen2-VL / Qwen2.5-VL derivatives, so they share the chat-template + image
input conventions and reuse the Qwen wrapper's predict path. Each subclass
overrides only what differs (prompt template, action format).

Ferret-UI Lite checkpoint availability is not yet confirmed at the time of
writing; the spec is a placeholder until Apple publishes weights.
"""

from __future__ import annotations

import re
from typing import Any

from PIL.Image import Image

from ..prompt.action import ParsedAction, parse_click
from ..prompt.templates import format_click_prompt
from .base import ModelOutput, TorchModel


class _QwenLikeSpecialist(TorchModel):
    """Shared loading + predict for Qwen2-VL-derived specialists."""

    family = "specialist"
    default_prompt: str | None = None

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 64,
        quant_config: Any = None,
        **load_kwargs: Any,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.param_count_b = _infer_param_count(model_id)
        self.max_new_tokens = max_new_tokens

        kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(torch_dtype),
            "device_map": device_map,
            "trust_remote_code": True,
            **load_kwargs,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        # ShowUI is Qwen2-VL-2B based; Qwen2VLConfig is NOT in the CausalLM auto-map,
        # so use the image-text-to-text auto class (-> Qwen2VLForConditionalGeneration).
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _build_messages(self, image: Image, instruction: str) -> list[dict]:
        prompt = format_click_prompt(
            instruction,
            **({"template": self.default_prompt} if self.default_prompt else {}),
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def predict(
        self,
        image: Image,
        instruction: str,
        **gen_kwargs: Any,
    ) -> ModelOutput:
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(image, instruction)
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            **gen_kwargs,
        )
        generated = gen[:, inputs.input_ids.shape[1] :]
        response: str = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return ModelOutput(
            text=response,
            parsed=parse_click(response, image_size=image.size),
            metadata={"prompt": chat_text, "model_id": self.model_id},
        )


# ── OS-Atlas-Pro-4B (InternVL2: InternViT + Phi-3) ────────────────────────────
# OS-Atlas-Pro-4B is NOT a Qwen2-VL derivative. It is an InternVL2-4B *action*
# model: tiled 448px vision, a `model.chat()` API (no AutoProcessor/chat-template
# the Qwen way), and a system prompt that elicits "Thoughts:" + "Actions: CLICK
# <point>[[x, y]]</point>" with coordinates normalized to [0, 1000].

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Exact system prompt from the OS-Atlas-Pro-4B model card. {instruction} is
# substituted via str.replace (not .format) so the literal brackets in the
# action examples can't trip format parsing.
_OSATLAS_SYS_PROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability.
Basic Action 1: CLICK
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>

Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]

2.Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users.
Custom Action 1: LONG_PRESS
    - purpose: Long press at the specified position.
    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_PRESS <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating.

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot:
<image>
Task instruction: {instruction}
History: null
"""

# "CLICK <point>[[x, y]]</point>" (preferred) and bare "[[x, y]]" fallback.
_OSATLAS_POINT_RE = re.compile(
    r"<point>\s*\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]"
)
_OSATLAS_BARE_RE = re.compile(
    r"\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]"
)


def _patch_internvl_config(config_cls: type) -> None:
    """Make OS-Atlas's remote InternVLChatConfig survive no-arg construction.

    The card's config does `llm_config['architectures'][0]`, but transformers'
    `to_diff_dict()` builds `self.__class__()` with no args, leaving
    `llm_config={}` -> KeyError. Inject a Phi-3 default on the empty path only;
    the real load (with a populated llm_config) is untouched.
    """
    if getattr(config_cls, "_ais5_patched", False):
        return
    orig_init = config_cls.__init__

    def _safe_init(self: Any, *args: Any, **kwargs: Any) -> None:
        lc = kwargs.get("llm_config")
        if not lc or "architectures" not in lc:
            kwargs["llm_config"] = {"architectures": ["Phi3ForCausalLM"]}
        orig_init(self, *args, **kwargs)

    config_cls.__init__ = _safe_init
    config_cls._ais5_patched = True


def _ensure_generation_mixin(module: Any) -> None:
    """Re-grant `.generate` to a remote LM that lost GenerationMixin.

    transformers>=4.50 stopped having PreTrainedModel inherit GenerationMixin, so
    OS-Atlas's vendored Phi3ForCausalLM (which only defines
    prepare_inputs_for_generation) loses `.generate`. InternVLChatModel.chat()
    calls self.language_model.generate, so mix it back in on the instance.
    """
    if module is None:
        return
    from transformers import GenerationConfig
    from transformers.generation import GenerationMixin

    if not isinstance(module, GenerationMixin):
        module.__class__ = type(
            f"{module.__class__.__name__}WithGeneration",
            (module.__class__, GenerationMixin),
            {},
        )
    # GenerationMixin.generate reads self.generation_config; the vendored LM never
    # set one, so build it from the model config (gets eos/pad tokens right).
    if getattr(module, "generation_config", None) is None:
        try:
            module.generation_config = GenerationConfig.from_model_config(module.config)
        except Exception:  # noqa: BLE001
            module.generation_config = GenerationConfig()


def _patch_dynamic_cache() -> None:
    """OS-Atlas's vendored Phi-3 reads `past_key_values.seen_tokens`, which
    transformers>=~4.47 removed from DynamicCache. Map it to get_seq_length()."""
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:  # noqa: BLE001
        return
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        # Removed ~4.48 (renamed get_max_cache_shape); DynamicCache is unbounded.
        def _get_max_length(self):  # noqa: ANN001
            fn = getattr(self, "get_max_cache_shape", None)
            return fn() if fn is not None else None

        DynamicCache.get_max_length = _get_max_length
    if not hasattr(DynamicCache, "get_usable_length"):
        # Removed ~4.48. Unbounded cache -> already-cached length for that layer.
        def _get_usable_length(self, new_seq_length, layer_idx=0):  # noqa: ANN001
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length


def _internvl_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _dynamic_preprocess(image: Image, *, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = True):
    """InternVL dynamic tiling: split into up to `max_num` 448px tiles + thumbnail."""
    w, h = image.size
    ar = w / h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    tar = _closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class OSAtlas(TorchModel):
    """OS-Atlas-Pro-4B — InternVL2 (InternViT + Phi-3) GUI action model (paper #2).

    Loaded via `AutoModel` + `model.chat` (the InternVL path), NOT the Qwen
    wrapper. It emits `Actions: CLICK <point>[[x, y]]</point>` with x,y in
    [0, 1000]; we denormalize to pixels so the shared click scorer applies.
    """

    family = "specialist"
    coord_norm: float | None = 1000.0  # set None to treat outputs as raw pixels

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 384,
        quant_config: Any = None,
        max_tiles: int = 6,
        **load_kwargs: Any,
    ) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.param_count_b = _infer_param_count(model_id)
        # OS-Atlas reasons ("Thoughts:") before the action, so it needs room to
        # reach the CLICK line; floor the budget regardless of the eval config's
        # specialist default (64) which would truncate before the action.
        self.max_new_tokens = max(int(max_new_tokens), 384)
        self.max_tiles = max_tiles

        # Patch the remote InternVLChatConfig BEFORE any config load. Both
        # AutoConfig and AutoModel eagerly f-string-log the config
        # (`logger.info(f"Model config {config}")` -> repr -> to_diff_dict ->
        # self.__class__()), which KeyErrors on the card's no-arg path. The
        # dynamic-module loader imports the class without instantiating it, so we
        # patch __init__ first; AutoConfig then reuses the same cached class.
        config_cls = get_class_from_dynamic_module(
            "configuration_internvl_chat.InternVLChatConfig", model_id
        )
        _patch_internvl_config(config_cls)
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(torch_dtype),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            **load_kwargs,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        self.model = AutoModel.from_pretrained(model_id, config=cfg, **kwargs).eval()
        _ensure_generation_mixin(getattr(self.model, "language_model", None))
        _patch_dynamic_cache()
        self.processor = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

    def _pixel_values(self, image: Image):
        import torch

        tiles = _dynamic_preprocess(image.convert("RGB"), max_num=self.max_tiles)
        transform = _internvl_transform(448)
        pv = torch.stack([transform(t) for t in tiles])
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        # bf16 is the model's compute dtype (config torch_dtype / bnb compute_dtype),
        # independent of any quantized weight dtype.
        return pv.to(device=device, dtype=torch.bfloat16)

    def predict(self, image: Image, instruction: str, **gen_kwargs: Any) -> ModelOutput:
        pixel_values = self._pixel_values(image)
        question = _OSATLAS_SYS_PROMPT.replace("{instruction}", instruction)
        gen_config: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        gen_config.update(gen_kwargs)
        response = self.model.chat(self.processor, pixel_values, question, gen_config)
        if isinstance(response, tuple):  # return_history=True would give (resp, hist)
            response = response[0]
        point = self._parse_point(response, image.size)
        return ModelOutput(
            text=response,
            parsed=ParsedAction(
                point=point, raw=response, parser="os-atlas-point" if point else "none"
            ),
            metadata={"model_id": self.model_id, "prompt": "os-atlas-action"},
        )

    def _parse_point(self, text: str, image_size: tuple[int, int]) -> tuple[float, float] | None:
        # The action is emitted after "Thoughts:", so take the LAST coordinate match.
        match = None
        for match in _OSATLAS_POINT_RE.finditer(text):
            pass
        if match is None:
            for match in _OSATLAS_BARE_RE.finditer(text):
                pass
        if match is None:
            return None
        x, y = float(match.group(1)), float(match.group(2))
        if self.coord_norm:
            w, h = image_size
            x = x / self.coord_norm * w
            y = y / self.coord_norm * h
        return (x, y)


class ShowUI(_QwenLikeSpecialist):
    """ShowUI-2B (paper #3, Qwen2-VL-2B based). Smallest specialist.

    ShowUI is trained with its own grounding system prompt and outputs a
    NORMALIZED point `[x, y]` in [0, 1] (not the project's `<click>`/pixel
    format). We use its native prompt and denormalize to pixels so the shared
    click scorer applies.
    """

    _GROUNDING_SYSTEM = (
        "Based on the screenshot of the page, I give a text description and you give its "
        "corresponding location. The coordinate represents a clickable location [x, y] "
        "for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
    )
    _XY_RE = re.compile(r"[\[(]\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\])]")

    def predict(self, image: Image, instruction: str, **gen_kwargs: Any) -> ModelOutput:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": self._GROUNDING_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, **gen_kwargs
        )
        generated = gen[:, inputs.input_ids.shape[1] :]
        response: str = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        point = self._parse_showui(response, image.size)
        return ModelOutput(
            text=response,
            parsed=ParsedAction(
                point=point, raw=response, parser="showui-xy" if point else "none"
            ),
            metadata={"model_id": self.model_id, "prompt": "showui-grounding"},
        )

    @classmethod
    def _parse_showui(cls, text: str, image_size: tuple[int, int]) -> tuple[float, float] | None:
        m = cls._XY_RE.search(text)
        if m is None:
            return None
        x, y = float(m.group(1)), float(m.group(2))
        w, h = image_size
        # ShowUI emits [0,1]-normalized coords; denormalize. Guard the rare case
        # where it returns pixel-scale values (>1.5) by treating those as pixels.
        if abs(x) <= 1.5 and abs(y) <= 1.5:
            x, y = x * w, y * h
        return (x, y)


class FerretUILite(_QwenLikeSpecialist):
    """Ferret-UI Lite-3B (paper #1) — primary specialist target.

    NOTE: As of the project proposal, weights may not yet be on the Hub. The
    `model_id` in the registry is a placeholder; update it once Apple releases.
    Until then, evaluating this row will fail at load time — that's the
    documented "Risk" branch in Task 3's backup plan.
    """

    default_prompt = (
        "You are a mobile UI agent. Locate the element described and click it. "
        "Output `<click>x, y</click>` in screen pixels.\n\nElement: {instruction}"
    )


def _infer_param_count(model_id: str) -> float:
    lower = model_id.lower()
    for size in ("8b", "7b", "4b", "3b", "2b", "1b"):
        if size in lower:
            return float(size.rstrip("b"))
    return 0.0


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            return torch.float32
        return "auto"
    return getattr(torch, name)
