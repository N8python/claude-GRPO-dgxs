#!/bin/bash
# Generic GRPO run + evals. RUN ON ALICE, with Bob serving the same model.
#
#   M=google/gemma-4-E4B-it ADAPTER=models/gemma-sft-lora OUT=models/gemma-grpo \
#   G=16 TAG=gemma-g16 setsid nohup ./run_grpo.sh > logs/gemma.log 2>&1 < /dev/null &
#
# run_qwen.sh is the Qwen-specific instance of this; kept for reproducibility.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-$HOME/miniforge3/envs/femtogpt-torch/bin/python}"
BOB="${BOB:-http://192.168.100.11:8300}"
M="${M:?set M to the model id}"
ADAPTER="${ADAPTER:?set ADAPTER to the SFT adapter dir}"
OUT="${OUT:?set OUT to the output adapter prefix}"
G="${G:-8}"
P="${P:-4}"
MB="${MB:-8}"
STEPS="${STEPS:-300}"
SEED="${SEED:-42}"
TAG="${TAG:-run}"
WBP="${WBP:-dgx-grpo-proper}"

mkdir -p logs
stamp(){ date +"%H:%M:%S"; }

echo "[$(stamp)] ===== GRPO $TAG: $M, $STEPS steps, P=$P G=$G microbatch=$MB, seed $SEED ====="
START=$(date +%s)
"$PY" train_grpo.py --model "$M" --rollout-url "$BOB" --adapter-path "$ADAPTER" \
  --steps "$STEPS" --seed "$SEED" --out "$OUT" \
  --prompts-per-step "$P" --group-size "$G" --microbatch "$MB" \
  --wandb-project "$WBP" --wandb-run-name "$TAG-seed$SEED" \
  > "logs/train-$TAG.log" 2>&1
RC=$?
END=$(date +%s)
echo "[$(stamp)] training exit=$RC  wall=$(( (END-START)/60 ))m$(( (END-START)%60 ))s  ($(grep -c '^step' "logs/train-$TAG.log") steps logged)"
grep '^step' "logs/train-$TAG.log" | tail -1

for ck in "$OUT-step100" "$OUT-step200" "$OUT"; do
  tag="$(basename "$ck")"
  [ -d "$ck" ] || { echo "[$(stamp)]   SKIP $tag"; continue; }
  "$PY" evaluate.py --model "$M" --rollout-url "$BOB" --adapter-path "$ck" \
    --temp 0.7 > "logs/eval-$tag.log" 2>&1
  echo "[$(stamp)]   $tag -> $(grep 'exact match' "logs/eval-$tag.log" | sed 's/exact match: *//')"
done

echo
echo "[$(stamp)] ==================== SUMMARY ===================="
printf "%-26s %-16s %s\n" "checkpoint" "exact @ t0.7" "avg lev"
for f in logs/eval-$(basename "$OUT")*.log; do
  [ -e "$f" ] || continue
  printf "%-26s %-16s %s\n" "$(basename "$f" .log | sed 's/^eval-//')" \
    "$(grep 'exact match' "$f" | sed 's/.*= *//')" \
    "$(grep 'levenshtein' "$f" | sed 's/.*: *//')"
done
echo
echo "train wall: $(( (END-START)/60 ))m$(( (END-START)%60 ))s   s/step: $(grep '^step' "logs/train-$TAG.log" | tail -1 | grep -o '([0-9.]* s/step' | tr -d '(')"
echo "[$(stamp)] done"
