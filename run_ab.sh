#!/bin/bash
# A/B: strict on-policy vs one-step-stale async. Same seed, same budget.
# RUN ON ALICE, with Bob's rollout server already up.
#
#   ssh Alice 'cd ~/dgxGRPOProper && setsid nohup ./run_ab.sh > logs/ab.log 2>&1 < /dev/null &'
#
# Both arms train from the same SFT adapter and are evaluated at the training
# temperature (0.7) for an on-policy measurement, at steps 100/200/300 so the
# curves are comparable and not just the endpoints.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-$HOME/miniforge3/envs/femtogpt-torch/bin/python}"
BOB="${BOB:-http://192.168.100.11:8300}"
WBP="${WBP:-dgx-grpo-proper}"
STEPS="${STEPS:-300}"
SEED="${SEED:-42}"

mkdir -p logs
stamp(){ date +"%H:%M:%S"; }

train(){
  local name="$1"; shift
  echo "[$(stamp)] ===== GRPO $name: $STEPS steps, seed $SEED ====="
  "$PY" train_grpo.py --rollout-url "$BOB" --adapter-path models/sft-lora \
    --steps "$STEPS" --seed "$SEED" --out "models/grpo-$name" \
    --wandb-project "$WBP" --wandb-run-name "$name-seed$SEED" "$@" \
    > "logs/train-$name.log" 2>&1
  local rc=$?
  echo "[$(stamp)] $name training exit=$rc  ($(grep -c '^step' "logs/train-$name.log") steps logged)"
  tail -1 "logs/train-$name.log"
  return $rc
}

ev(){
  local ck="$1" tag
  tag="$(basename "$ck")"
  [ -d "$ck" ] || { echo "[$(stamp)]   SKIP $tag (not written)"; return; }
  "$PY" evaluate.py --rollout-url "$BOB" --adapter-path "$ck" --temp 0.7 \
    > "logs/eval-$tag.log" 2>&1
  echo "[$(stamp)]   $tag -> $(grep 'exact match' "logs/eval-$tag.log" | sed 's/exact match: *//')"
}

for arm in strict async; do
  if [ "$arm" = async ]; then train async; else train strict --no-async-rollouts; fi
  echo "[$(stamp)] --- $arm evals (temp 0.7) ---"
  for s in 100 200; do ev "models/grpo-$arm-step$s"; done
  ev "models/grpo-$arm"
done

echo
echo "[$(stamp)] ==================== SUMMARY ===================="
echo "SFT baseline, greedy: 34.3% exact / lev 1.47   (MLX reference: 34.5 / 1.49)"
echo "MLX reference GRPO 300 steps @ temp 0.7: 51.6% exact"
echo
printf "%-26s %-16s %s\n" "checkpoint" "exact" "avg lev"
for f in logs/eval-grpo-*.log; do
  [ -e "$f" ] || continue
  printf "%-26s %-16s %s\n" \
    "$(basename "$f" .log | sed 's/^eval-//')" \
    "$(grep 'exact match' "$f" | sed 's/.*= *//')" \
    "$(grep 'levenshtein' "$f" | sed 's/.*: *//')"
done
echo
for arm in strict async; do
  echo "$arm: $(grep '^step' "logs/train-$arm.log" | tail -1 | grep -o '([0-9.]* s/step' | tr -d '(')"
done
echo "[$(stamp)] done"
