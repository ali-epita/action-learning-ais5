# syntax=docker/dockerfile:1.7
#
# AIS 5 — Small VLMs vs. GUI Specialists
#
# Multi-stage build:
#   1. builder: uv image, resolves and installs deps into /opt/venv
#   2. runtime: slim Python, copies the prebuilt venv
#
# CPU build by default. For CUDA, build with:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 .

ARG PYTHON_VERSION=3.11

# ── builder ───────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first so they're cached independently of source changes.
COPY pyproject.toml ./
COPY uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --extra notebook

# Now install the project itself.
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra notebook

# ── runtime ───────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Minimal extras: git for HF Hub clones, curl for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/.hf_cache \
    HF_DATASETS_CACHE=/data/.hf_cache/datasets \
    TRANSFORMERS_CACHE=/data/.hf_cache/transformers \
    PYTHONPATH=/app/src

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY notebooks ./notebooks
COPY tests ./tests
COPY pyproject.toml README.md ./

# Default sanity-check command. Compose overrides this.
EXPOSE 8888
CMD ["python", "-c", "import ais5; print(f'ais5 v{ais5.__version__} ready')"]
