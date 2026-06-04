#!/usr/bin/env bash
set -euo pipefail

REPO="${1:?repo path is required}"
MODE="${2:-train}"

UV_BIN="${UV_BIN:-/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/general/nfs1/u1320844}"
FLASH_ROOT="${FLASH_ROOT:-$CACHE_ROOT/uv-envs}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.0}"
MAX_JOBS="${MAX_JOBS:-8}"

ensure_module_command() {
  if type module >/dev/null 2>&1; then
    return 0
  fi
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [ -f /usr/share/Modules/init/bash ]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
  type module >/dev/null 2>&1
}

ensure_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  if ensure_module_command; then
    module load "$CUDA_MODULE" >/dev/null 2>&1 || true
  fi
  command -v nvcc >/dev/null 2>&1
}

if [ ! -x "$UV_BIN" ]; then
  echo "uv not found at $UV_BIN" >&2
  exit 1
fi

mkdir -p "$FLASH_ROOT"

BASE_ENV="$FLASH_ROOT/kabr-base"
UV_PROJECT_ENVIRONMENT="$BASE_ENV" "$UV_BIN" sync --project "$REPO" --extra train --extra infer >/dev/null
BASE_PYTHON="$BASE_ENV/bin/python"

GPU_CAPABILITY="$("$BASE_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to prepare a flash-attn environment.")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"

FLASH_ENV="$FLASH_ROOT/kabr-flash-cc${GPU_CAPABILITY//./}"
UV_PROJECT_ENVIRONMENT="$FLASH_ENV" "$UV_BIN" sync --project "$REPO" --extra train --extra infer >/dev/null
FLASH_PYTHON="$FLASH_ENV/bin/python"

if ! "$FLASH_PYTHON" - <<'PY' >/dev/null 2>&1
import flash_attn  # noqa: F401
PY
then
  if ! ensure_nvcc; then
    echo "FlashAttention requested, but nvcc is unavailable. Load a CUDA toolkit before running." >&2
    exit 1
  fi
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  export TORCH_CUDA_ARCH_LIST="$GPU_CAPABILITY"
  export MAX_JOBS
  "$UV_BIN" pip install --python "$FLASH_PYTHON" --no-build-isolation flash-attn
fi

if ! "$FLASH_PYTHON" - <<'PY' >/dev/null 2>&1
import flash_attn  # noqa: F401
PY
then
  echo "flash_attn installation did not succeed for $FLASH_ENV" >&2
  exit 1
fi

printf '%s\n' "$FLASH_PYTHON"
