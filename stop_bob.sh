#!/bin/bash
# Stop the rollout server. RUN THIS ON BOB.
#
# Killing the HTTP process alone is not enough: vLLM's EngineCore is a
# multiprocessing child that survives and keeps holding the GPU pool, which
# then starves the next launch. Kill both.
#
# The bracket patterns are load-bearing. `pkill -f rollout_server.py` invoked
# over ssh matches the ssh command line itself and kills the session.
set -uo pipefail

echo "before:"
pgrep -af "[r]ollout_server.py" || echo "  (no rollout_server)"
pgrep -af "[V]LLM::EngineCore"  || echo "  (no EngineCore)"

pkill -f "[r]ollout_server.py" 2>/dev/null
sleep 3
pkill -f "[V]LLM::EngineCore" 2>/dev/null
sleep 2
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null

echo "after:"
pgrep -af "[r]ollout_server.py" || echo "  rollout_server: clear"
pgrep -af "[V]LLM::EngineCore"  || echo "  EngineCore: clear"
echo
echo "GPU compute apps still resident:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
  || echo "  (nvidia-smi unavailable)"
