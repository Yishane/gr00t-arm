#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the arm's dataset (run on the cloud GPU machine).
#   bash finetune.sh ~/pick_ball            # dataset folder copied from data/lerobot/pick_ball
# Optional: MAX_STEPS=3000 BATCH=32 bash finetune.sh ~/pick_ball
# Checkpoints land in ~/gr00t_arm_ckpt/checkpoint-<step>.
set -euo pipefail
DATASET="$(realpath "${1:?usage: finetune.sh <dataset folder>}")"
HERE="$(cd "$(dirname "$0")" && pwd)"
MAX_STEPS="${MAX_STEPS:-3000}"
BATCH="${BATCH:-32}"
OUT="${OUT:-$HOME/gr00t_arm_ckpt}"
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/Isaac-GR00T"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$HERE/gr00t_arm_config.py" \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --max-steps "$MAX_STEPS" \
  --save-steps 1000 \
  --global-batch-size "$BATCH" \
  --dataloader-num-workers 4

# sanity check: does the model reproduce episode 0? (plot in /tmp/open_loop_eval)
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "$OUT/checkpoint-$MAX_STEPS" \
  --traj-ids 0 --execution-horizon 16 --steps 250 || true
echo "done: $OUT/checkpoint-$MAX_STEPS"
