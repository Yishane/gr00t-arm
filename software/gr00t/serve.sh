#!/usr/bin/env bash
# Serve the fine-tuned model so the laptop can drive the arm with it.
#   bash serve.sh                         # newest checkpoint in ~/gr00t_arm_ckpt
#   bash serve.sh ~/gr00t_arm_ckpt/checkpoint-3000
# Then on the laptop:  uv run run_policy.py gr00t --host <this machine's IP> --server-port 5555
# (open TCP port 5555 in the cloud provider's firewall / port settings, or use an SSH
#  tunnel: ssh -L 5555:localhost:5555 <user>@<ip>  and connect to --host localhost)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-$(ls -d "$HOME"/gr00t_arm_ckpt/checkpoint-* | sort -V | tail -1)}"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/Isaac-GR00T"
echo "serving $CKPT on port 5555"
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$HERE/gr00t_arm_config.py" \
  --device cuda:0 --host 0.0.0.0 --port 5555
