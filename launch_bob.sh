#!/bin/bash
# Start the vLLM rollout server. RUN THIS ON BOB.
#
# Foreground by default so a supervisor can wrap it. To detach over ssh, use
# setsid+nohup with stdin redirected -- a plain `nohup cmd &` hangs the ssh
# session on these boxes:
#
#   ssh Bob 'cd ~/dgxGRPOProper && setsid nohup ./launch_bob.sh \
#            > bob_server.log 2>&1 < /dev/null &'
#
# Override the interpreter if the vLLM env is elsewhere:
#   PY=~/miniforge3/envs/<env>/bin/python ./launch_bob.sh
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"
MODEL="${MODEL:-mlx-community/Llama-3.2-1B-Instruct-bf16}"
PORT="${PORT:-8300}"
# Explicit and low. vLLM's ~0.9 default would pre-allocate ~99 GB of GB10's
# 121 GB unified pool for a model that needs a few.
GPU_UTIL="${GPU_UTIL:-0.35}"
MAX_LEN="${MAX_LEN:-1024}"
MAX_SEQS="${MAX_SEQS:-64}"

# The vLLM worker runs in its own process and must import vllm_weight_sync.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[bob] $($PY -c 'import vllm; print("vllm", vllm.__version__)' 2>/dev/null || echo 'vllm: NOT FOUND')"
echo "[bob] model=$MODEL port=$PORT gpu_util=$GPU_UTIL max_len=$MAX_LEN"

exec "$PY" rollout_server.py \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --logprobs-mode processed_logprobs \
  "$@"
