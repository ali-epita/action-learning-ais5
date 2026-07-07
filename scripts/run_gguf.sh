#!/usr/bin/env bash
# Convert the merged Qwen2.5-VL (our LoRA-r64) to cross-platform GGUF Q4 (+ vision mmproj)
# via llama.cpp. CPU-only; runs alongside the GPU demo. Non-fatal per step.
set +e
MERGED=/workspace/merged-qwen-r64
OUT=/workspace/gguf
mkdir -p "$OUT"
PY=/workspace/.venv-ais5/bin/python
cd /workspace/llama.cpp || { echo "llama.cpp missing"; echo "EXIT=no-llamacpp" > /workspace/gguf.status; exit 1; }

echo ">>> [1/5] convert deps"
uv pip install --python /workspace/.venv-ais5 -q gguf sentencepiece protobuf mistral-common 2>&1 | tail -1

echo ">>> [2/5] text model -> GGUF f16"
$PY convert_hf_to_gguf.py "$MERGED" --outfile "$OUT/qwen-r64-f16.gguf" --outtype f16 2>&1 | tail -8

echo ">>> [3/5] vision encoder -> mmproj"
$PY convert_hf_to_gguf.py "$MERGED" --mmproj --outfile "$OUT/qwen-r64-mmproj-f16.gguf" 2>&1 | tail -8

echo ">>> [4/5] build llama-quantize (CPU)"
if [ ! -x build/bin/llama-quantize ]; then
  cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /tmp/cmake.log 2>&1 && \
  cmake --build build --target llama-quantize -j4 > /tmp/build.log 2>&1 && echo "  build ok" || { echo "  build FAILED"; tail -6 /tmp/build.log; }
fi

echo ">>> [5/5] quantize -> Q4_K_M"
if [ -f "$OUT/qwen-r64-f16.gguf" ] && [ -x build/bin/llama-quantize ]; then
  ./build/bin/llama-quantize "$OUT/qwen-r64-f16.gguf" "$OUT/qwen-r64-q4_k_m.gguf" Q4_K_M 2>&1 | tail -4
fi

echo "=== artifacts ==="
ls -la "$OUT"/*.gguf 2>/dev/null
echo "EXIT=$?" > /workspace/gguf.status
