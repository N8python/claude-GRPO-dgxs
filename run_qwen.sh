#!/bin/bash
# Qwen3-4B-Instruct-2507 GRPO run. RUN ON ALICE, Bob serving the same model.
#
#   ssh Bob   'cd ~/dgxGRPOProper && MODEL=Qwen/Qwen3-4B-Instruct-2507 GPU_UTIL=0.35 \
#              PY=~/miniforge3/envs/gemmacev/bin/python setsid nohup ./launch_bob.sh \
#              > bob_qwen.log 2>&1 < /dev/null &'
#   ssh Alice 'cd ~/dgxGRPOProper && setsid nohup ./run_qwen.sh > logs/qwen.log 2>&1 < /dev/null &'
#
# Assumes models/qwen-sft-lora already exists (train_sft.py --model <M>).
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-$HOME/miniforge3/envs/femtogpt-torch/bin/python}"
BOB="${BOB:-http://192.168.100.11:8300}"
M="${M:-Qwen/Qwen3-4B-Instruct-2507}"
STEPS="${STEPS:-300}"
SEED="${SEED:-42}"
WBP="${WBP:-dgx-grpo-proper}"

mkdir -p logs
stamp(){ date +"%H:%M:%S"; }

echo "[$(stamp)] ===== Qwen3-4B GRPO: $STEPS steps, seed $SEED (pipelined default) ====="
START=$(date +%s)
"$PY" train_grpo.py --model "$M" --rollout-url "$BOB" \
  --adapter-path models/qwen-sft-lora \
  --steps "$STEPS" --seed "$SEED" --out models/qwen-grpo \
  --wandb-project "$WBP" --wandb-run-name "qwen3-4b-seed$SEED" \
  > logs/train-qwen.log 2>&1
RC=$?
END=$(date +%s)
echo "[$(stamp)] training exit=$RC  wall=$(( (END-START)/60 ))m$(( (END-START)%60 ))s  ($(grep -c '^step' logs/train-qwen.log) steps logged)"
grep '^step' logs/train-qwen.log | tail -1

for ck in models/qwen-grpo-step100 models/qwen-grpo-step200 models/qwen-grpo; do
  tag="$(basename "$ck")"
  [ -d "$ck" ] || { echo "[$(stamp)]   SKIP $tag"; continue; }
  "$PY" evaluate.py --model "$M" --rollout-url "$BOB" --adapter-path "$ck" \
    --temp 0.7 > "logs/eval-$tag.log" 2>&1
  echo "[$(stamp)]   $tag -> $(grep 'exact match' "logs/eval-$tag.log" | sed 's/exact match: *//')"
done

echo
echo "[$(stamp)] ==================== SUMMARY ===================="
printf "%-24s %-16s %s\n" "checkpoint" "exact @ t0.7" "avg lev"
for f in logs/eval-qwen-grpo*.log; do
  [ -e "$f" ] || continue
  printf "%-24s %-16s %s\n" "$(basename "$f" .log | sed 's/^eval-//')" \
    "$(grep 'exact match' "$f" | sed 's/.*= *//')" \
    "$(grep 'levenshtein' "$f" | sed 's/.*: *//')"
done
echo
echo "train wall: $(( (END-START)/60 ))m$(( (END-START)%60 ))s   s/step: $(grep '^step' logs/train-qwen.log | tail -1 | grep -o '([0-9.]* s/step' | tr -d '(')"
echo "[$(stamp)] done"
