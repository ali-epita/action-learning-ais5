.DEFAULT_GOAL := help
SHELL := /bin/bash

# ──────────────────────────────────────────────────────────────────────────────
# AIS 5 — convenience targets. All commands run via `uv run` so the venv is
# always honored, no manual `source .venv/bin/activate` needed.
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	      /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } \
	      /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup
.PHONY: sync sync-quant sync-all lock
sync: ## install runtime + dev deps via uv
	uv sync

sync-quant: ## include quantization extras (Linux/CUDA only)
	uv sync --extra quant --extra notebook

sync-all: ## install everything optional too
	uv sync --all-extras

lock: ## refresh uv.lock without installing
	uv lock

##@ Code quality
.PHONY: lint fmt test
lint: ## ruff check
	uv run ruff check src tests scripts

fmt: ## ruff fix + format
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

test: ## pytest, skipping anything tagged "gpu" or "remote"
	uv run pytest -m "not gpu and not remote"

test-all: ## pytest including gpu/remote tests
	uv run pytest

##@ Run
.PHONY: eval bench train papers shell
eval: ## zero-shot eval on Qwen2.5-VL → ScreenSpot-V2
	uv run python -m ais5.cli eval --config configs/eval/zero_shot_qwen.yaml

bench: ## efficiency benchmark across models
	uv run python -m ais5.cli bench --config configs/bench/efficiency.yaml

train: ## LoRA fine-tune Qwen2.5-VL on OS-Atlas + UGround subset
	uv run python -m ais5.cli train --config configs/train/lora_qwen.yaml

papers: ## download bibliography PDFs into ../Papers
	uv run python scripts/download_papers.py --out "../Papers"

shell: ## drop into an interactive Python shell with ais5 pre-imported
	uv run python -c "import ais5, IPython; IPython.embed()" 2>/dev/null \
		|| uv run python -i -c "import ais5; print(f'ais5 v{ais5.__version__} loaded')"

##@ Docker
.PHONY: docker-build docker-up docker-down docker-shell jupyter
docker-build: ## build image
	docker compose --profile dev --profile jupyter build

docker-up: ## start dev container in background
	docker compose --profile dev up -d dev

docker-shell: ## bash inside the dev container
	docker compose --profile dev exec dev bash

jupyter: ## start Jupyter Lab on http://localhost:8888
	docker compose --profile jupyter up -d jupyter
	@echo
	@echo "  Jupyter Lab → http://localhost:8888"
	@echo "  (no token, no password — local-only by default)"
	@echo

docker-down: ## stop and remove all ais5 containers
	docker compose --profile dev --profile jupyter down

##@ Housekeeping
.PHONY: clean clean-all
clean: ## remove caches and build artefacts
	rm -rf .venv .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

clean-all: clean ## also remove downloaded data and results
	rm -rf data results checkpoints wandb
