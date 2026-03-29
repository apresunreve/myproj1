#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-.uvenv/bin/python}"

"$UV_BIN" run --python "$PYTHON_BIN" python -m kabr_vlm.training.finetune --config configs/train.example.yaml
