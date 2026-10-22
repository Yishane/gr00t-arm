#!/usr/bin/env bash
# One-time setup on a rented Linux GPU machine (Ubuntu 22.04/24.04, NVIDIA GPU
# with >= 40 GB for fine-tuning, e.g. L40S / A100 / H100).
#   export HF_TOKEN=hf_xxx            # your Hugging Face token (read access)
#   bash setup_cloud.sh
# Your HF account must have been granted access to nvidia/Cosmos-Reason2-2B.
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "set HF_TOKEN first: export HF_TOKEN=hf_..." >&2
  exit 1
fi

sudo apt-get update -y
# GR00T's video decoder (torchcodec) needs FFmpeg 4-7; Ubuntu 22.04/24.04 ship 4.4 / 6.1
sudo apt-get install -y git git-lfs ffmpeg tmux
git lfs install

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME"
[ -d Isaac-GR00T ] || git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
uv sync --python 3.12
uv run python -c "import gr00t; print('GR00T installed')"
uv run hf auth login --token "$HF_TOKEN"
# pull the base model now so the first training run doesn't stall on it
uv run hf download nvidia/GR00T-N1.7-3B >/dev/null
uv run hf download nvidia/Cosmos-Reason2-2B >/dev/null
echo "setup done. Next: copy your dataset up and run finetune.sh"
