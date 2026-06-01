#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

POP=128
GENS=500
ALGO=open_es
SEEDS="42 123 7"
TMPDIR=/tmp/evosax_big

mkdir -p "$TMPDIR"

for SEED in $SEEDS; do
  OUT="runs/${ALGO}_p${POP}g${GENS}_seed${SEED}.npz"
  if [ -f "$OUT" ]; then
    echo "SKIP $OUT"
    continue
  fi
  echo "=== $ALGO pop=$POP gens=$GENS seed=$SEED ==="
  python neuroevolve_evosax.py \
    --algo "$ALGO" --seed "$SEED" \
    --pop "$POP" --gens "$GENS" \
    --out_dir "$TMPDIR"
  mv "$TMPDIR/${ALGO}_seed${SEED}.npz" "$OUT"
  echo "saved → $OUT"
done
echo "ALL DONE"
