#!/bin/sh
# Round-2 launcher for the compute server.
#
#   sh experiments/launch_round2.sh smoke    # check the GPT-5.4 configuration (~1 min)
#   sh experiments/launch_round2.sh all      # smoke, then start all shards detached
#   sh experiments/launch_round2.sh status   # progress of every shard
#
# Each shard is resumable: if one dies (or the box reboots), run `all` again and
# only the missing runs are repeated. Results: experiments/results/round2/.
# Credentials come from a shell file outside the repo (default ~/.gpt54_env,
# chmod 600; override with GPT_ENV_FILE) that exports OPENAI_API_KEY,
# OPENAI_BASE_URL and optionally GPT_MODEL. They are never written to logs or
# results.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/experiments/results/round2"

. "${GPT_ENV_FILE:-$HOME/.gpt54_env}"
export LLM_PROVIDER=openai
export LLM_MODEL="${GPT_MODEL:-gpt-5.4}"
export PYTHONHASHSEED=0             # builtin hash() feeds BIC's Hilbert index
export PYTHONUNBUFFERED=1
# Reasoning effort: "none" by default (short factual answers; hidden reasoning
# tokens would eat the 150-token budget and leave empty answers). If the
# deployment rejects it, `all` falls back to "low" once. The value actually
# used is recorded in every result record and in the meta files.
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-none}"

DRIVER="$ROOT/experiments/run_round2.py"

start() {   # start NAME ARGS...  (detached, survives SSH drops)
    name=$1; shift
    nohup setsid "$PY" -u "$DRIVER" "$@" > "$LOGS/round2_$name.log" 2>&1 < /dev/null &
    echo "$name $!" >> "$LOGS/round2.pids"
    echo "  started $name (pid $!)"
}

case "${1:-}" in
  smoke)
    echo "reasoning_effort=$LLM_REASONING_EFFORT"
    LLM_MAX_RETRIES=1 "$PY" -u "$DRIVER" smoke 2>&1 | tee "$LOGS/round2_smoke.log"
    ;;
  all)
    if ! LLM_MAX_RETRIES=1 "$PY" -u "$DRIVER" smoke > "$LOGS/round2_smoke.log" 2>&1; then
        tail -3 "$LOGS/round2_smoke.log"
        if [ "$LLM_REASONING_EFFORT" = none ]; then
            echo "smoke failed with reasoning_effort=none; retrying with low"
            export LLM_REASONING_EFFORT=low
            if ! LLM_MAX_RETRIES=1 "$PY" -u "$DRIVER" smoke > "$LOGS/round2_smoke.log" 2>&1; then
                tail -5 "$LOGS/round2_smoke.log"; echo "SMOKE FAILED, nothing launched"; exit 1
            fi
        else
            echo "SMOKE FAILED, nothing launched"; exit 1
        fi
    fi
    echo "smoke OK with reasoning_effort=$LLM_REASONING_EFFORT"
    : > "$LOGS/round2.pids"
    start sweep_c8     sweep --caches 8
    start sweep_c16    sweep --caches 16 --no-unbounded
    start sweep_c32    sweep --caches 32 --no-unbounded
    start hash         hash
    start pilot        pilot
    start order_bfs    order --orders bfs
    start order_dfs    order --orders dfs
    start order_random order --orders random
    echo "ROUND2_LAUNCHED_OK"
    ;;
  status)
    for f in "$LOGS"/round2_*.log; do
        echo "== $(basename "$f")"; grep -E "^\s*\[|finished|Error|Traceback|REJECTED" "$f" | tail -3
    done
    echo "== valid records per file"
    wc -l "$ROOT"/experiments/results/round2/*.jsonl 2>/dev/null
    echo "== live processes"
    while read -r name pid; do
        if kill -0 "$pid" 2>/dev/null; then echo "  $name running"; else echo "  $name exited"; fi
    done < "$LOGS/round2.pids"
    ;;
  *)
    echo "usage: $0 smoke|all|status"; exit 2
    ;;
esac
