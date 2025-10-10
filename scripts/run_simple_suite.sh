#!/usr/bin/env bash
set -euo pipefail

# Entorno
arch -x86_64 zsh -c '
source ~/venvs/dropwl_x86/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SEEDS="20250925 20250926 20250927 20250928 20250929"
OUTROOT="results/simple_suite"
mkdir -p "$OUTROOT"

declare -a JOBS=(
  # graph-level
  "data_synth/c4_n24.npz    c4"
  "data_synth/skip_n24.npz  skip"
  # node-level
  "data_synth/tri_nodes_n32.npz triangles"
  "data_synth/lcc_nodes_n64.npz lcc"
)

K=${#JOBS[@]}
i=0
for line in "${JOBS[@]}"; do
  DATA=$(echo $line | awk "{print \$1}")
  NAME=$(echo $line | awk "{print \$2}")
  OUT="$OUTROOT/$NAME"
  mkdir -p "$OUT"
  i=$((i+1))
  echo "[${i}/${K}] Ejecutando ${NAME} -> ${OUT}"

  PYTHONPATH=. python scripts/exp_simple_compare.py \
    --data "$DATA" \
    --seeds $SEEDS \
    --standardize \
    --wl_t 3 --kmax 32 \
    --R 50 \
    --layers 2 --hidden 128 --d 64 --act relu \
    --epochs 30 --lr 1e-3 --patience 8 \
    --outdir "$OUT"
done

echo "[OK] Suite simple terminada. Resultados en $OUTROOT"
'
