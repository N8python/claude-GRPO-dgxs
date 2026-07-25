#!/bin/bash
# Full reference pipeline. RUN THIS ON ALICE, with Bob's server already up.
#
#   Bob:    ./launch_bob.sh
#   Alice:  ./run_reference.sh
#
# SFT -> eval -> sync check -> GRPO 300 steps -> eval.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"
BOB="${BOB:-http://192.168.100.11:8300}"

echo "=== SFT (100 traces, rank-8 LoRA) ==="
"$PY" train_sft.py

echo "=== SFT eval (greedy) ==="
"$PY" evaluate.py --rollout-url "$BOB" --adapter-path models/sft-lora \
  --out preds_sft.jsonl

# Cheap, and the only thing that distinguishes "GRPO is training" from "GRPO
# is training on someone else's samples". Do not skip it.
echo "=== Alice<->Bob sync check ==="
"$PY" check_sync.py --rollout-url "$BOB" --adapter-path models/sft-lora

echo "=== GRPO (300 steps, TIS on) ==="
"$PY" train_grpo.py --rollout-url "$BOB" --adapter-path models/sft-lora

echo "=== GRPO eval (on-policy, temp 0.7) ==="
"$PY" evaluate.py --rollout-url "$BOB" --adapter-path models/grpo-lora \
  --temp 0.7 --out preds_grpo.jsonl

echo "=== done ==="
