#!/usr/bin/env bash
# The published runs end to end on one 48 GB GPU: the untrained model on the bench, then at each
# seed the task-only agent, the gated agent and the oracle gate (a reference), sixty steps each,
# each benched; then the results table and the play page. About six hours on an A40.
# Pass a list of seeds to run fewer: bash scripts/reproduce.sh 0 1
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${*:-0 1 2 3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/bench runs

[ -f results/bench/base/draw2.jsonl ] || uv run python scripts/bench.py --out results/bench/base
for s in $SEEDS; do
  for gate in none base oracle; do
    name="${gate}_s${s}"
    [ -f "runs/$name/adapter/adapter_model.safetensors" ] || \
      uv run python scripts/train.py --seed "$s" --gate "$gate" --name "$name"
    [ -f "results/bench/$name/draw2.jsonl" ] || \
      uv run python scripts/bench.py --adapter "runs/$name/adapter" --out "results/bench/$name"
  done
done
uv run python scripts/results.py
uv run python scripts/make_play.py --base results/bench/base/draw1.jsonl
