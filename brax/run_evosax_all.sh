#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

GENS=80
POP=50
SEEDS="42 123 7"
ALGOS="open_es cma_es sep_cma_es simple_ga"

for ALGO in $ALGOS; do
  for SEED in $SEEDS; do
    OUT="runs/${ALGO}_seed${SEED}.npz"
    if [ -f "$OUT" ]; then
      echo "SKIP $OUT (already exists)"
      continue
    fi
    echo "=== $ALGO seed=$SEED ==="
    python neuroevolve_evosax.py \
      --algo "$ALGO" --seed "$SEED" \
      --gens "$GENS" --pop "$POP" \
      --out_dir runs
  done
done
echo "ALL DONE"
