#!/usr/bin/env python3
"""Interactive screen-grounding demo: upload a UI screenshot + an instruction,
the model predicts where to click and draws it.

    python scripts/demo_app.py                                            # base Qwen2.5-VL-3B
    python scripts/demo_app.py --adapter checkpoints/qwen2.5-vl-3b-lora-r32  # our LoRA
    python scripts/demo_app.py --share                                    # public gradio.live URL

Runs anywhere the model loads (pod GPU, or locally on CPU/MPS — slower).
"""

from __future__ import annotations

import argparse

import gradio as gr
from PIL import Image, ImageDraw

from ais5.models import get_model


def build(model, model_label: str):
    def predict(image: Image.Image, instruction: str):
        if image is None or not (instruction or "").strip():
            return None, "Upload a screenshot and type an instruction."
        out = model.predict(image, instruction)
        pt = out.parsed.point
        img = image.convert("RGB").copy()
        d = ImageDraw.Draw(img)
        if pt is not None:
            x, y = pt
            r = max(12, int(min(img.size) * 0.02))
            d.ellipse([x - r, y - r, x + r, y + r], outline="#16a34a", width=5)
            d.line([x - r - 10, y, x + r + 10, y], fill="#16a34a", width=3)
            d.line([x, y - r - 10, x, y + r + 10], fill="#16a34a", width=3)
            status = f"✅ predicted click at (x={x:.0f}, y={y:.0f})"
        else:
            status = "⚠️ no click point parsed from the model output"
        return img, f"{status}\n\nraw model output:\n{(out.text or '')[:300]}"

    with gr.Blocks(title="GUI grounding demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# 🖱️ Screen-grounding demo — {model_label}\n"
            "Upload a UI screenshot, type an instruction (e.g. *“click the Settings icon”*, "
            "*“open the search bar”*), and the model predicts **where to click**. "
            "The green crosshair is the predicted point.\n\n"
            "*AIS 5 — Small VLMs vs. GUI Specialists.*"
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
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-3b")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter dir")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live URL")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()

    label = a.model + (f" + LoRA ({a.adapter.split('/')[-1]})" if a.adapter else " (zero-shot)")
    print(f"loading {label} ...")
    model = get_model(a.model, **({"peft_adapter": a.adapter} if a.adapter else {}))
    build(model, label).launch(share=a.share, server_name="0.0.0.0", server_port=a.port)


if __name__ == "__main__":
    main()
