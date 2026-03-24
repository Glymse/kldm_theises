#! /usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=/workspace/.venv

uv --version

uv sync --dev
uv pip install "setuptools>=70,<82" "omegaconf>=2.3,<2.4" "hydra-core>=1.3,<1.4" "emmet-core>=0.84,<0.85"

arch="$(uname -m)"
if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
  # MatterGen imports torch_scatter on this path, but prebuilt arm64 wheels are
  # not reliably available for our Torch version. Build it against the env's
  # installed torch instead of using isolated build envs.
  uv pip install --no-build-isolation --no-binary torch-scatter "torch-scatter==2.1.2"
fi

uv run python -m ipykernel install --user --name kldm --display-name "kldm (.venv)"
