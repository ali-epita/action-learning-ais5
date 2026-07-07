#!/usr/bin/env python3
"""Merge a LoRA adapter into the base Qwen2.5-VL and save a standalone model.

The adapter is a delta on the base weights, so to run it anywhere (esp. after
converting to MLX for the Mac) it has to be merged into full weights first.

    python scripts/merge_lora.py --adapter checkpoints/qwen2.5-vl-3b-lora-r64 --out merged-qwen-r64
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  base={a.base}  adapter={a.adapter}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.base, torch_dtype=torch.float16, device_map=dev
    )
    model = PeftModel.from_pretrained(base, a.adapter)
    print("merging adapter into base weights ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoProcessor.from_pretrained(a.base).save_pretrained(a.out)
    print(f"saved merged model -> {a.out}")


if __name__ == "__main__":
    main()
