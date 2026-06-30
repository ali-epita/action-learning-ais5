#!/usr/bin/env python3
"""Local screen-grounding demo on Apple Silicon via MLX — fully offline, no pod.

Runs our LoRA-r64 (85.5%) Qwen2.5-VL-3B, 4-bit MLX. Reuses the exact pod prompt +
click parser so behavior matches.

    .venv-demo/bin/python scripts/demo_mlx.py --model mlx-qwen-r64-4bit
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

import gradio as gr
from PIL import Image, ImageDraw
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

from ais5.prompt.action import parse_click
from ais5.prompt.templates import format_click_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-qwen-r64-4bit", help="path to the MLX 4-bit model dir")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    print(f"loading {a.model} ...")
    model, processor = load(a.model)
    try:
        config = model.config
    except Exception:
        from mlx_vlm.utils import load_config
        config = load_config(a.model)

    def predict(image: Image.Image, instruction: str):
        if image is None or not (instruction or "").strip():
            return None, "Upload a screenshot and type an instruction."
        img = image.convert("RGB")
        tmp = os.path.join(tempfile.gettempdir(), "mlx_demo_input.png")
        img.save(tmp)
        prompt = format_click_prompt(instruction)
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        res = generate(model, processor, formatted, image=[tmp], max_tokens=64, verbose=False)
        text = getattr(res, "text", str(res))
        pt = parse_click(text, image_size=img.size).point
        out = img.copy()
        d = ImageDraw.Draw(out)
        if pt is not None:
            x, y = pt
            r = max(12, int(min(img.size) * 0.02))
            d.ellipse([x - r, y - r, x + r, y + r], outline="#16a34a", width=5)
            d.line([x - r - 10, y, x + r + 10, y], fill="#16a34a", width=3)
            d.line([x, y - r - 10, x, y + r + 10], fill="#16a34a", width=3)
            status = f"✅ predicted click at ({x:.0f}, {y:.0f})"
        else:
            status = "⚠️ no click point parsed"
        return out, f"{status}\n\nraw model output:\n{text[:250]}"

    with gr.Blocks(title="GUI grounding — local MLX") as demo:
        gr.Markdown(
            "# 🖱️ Screen grounding — **local** (MLX, Apple Silicon)\n"
            "Our 85.5% Qwen2.5-VL-3B + LoRA, 4-bit — running **fully on your Mac, offline**. "
            "Upload a UI screenshot, type an instruction, get the predicted click."
        )
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="screenshot")
                instr = gr.Textbox(label="instruction", placeholder="click the search bar")
                btn = gr.Button("Predict click 🎯", variant="primary")
            with gr.Column():
                img_out = gr.Image(type="pil", label="predicted click")
                txt_out = gr.Textbox(label="model output", lines=6)
        btn.click(predict, [img_in, instr], [img_out, txt_out])
        instr.submit(predict, [img_in, instr], [img_out, txt_out])
    demo.launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
