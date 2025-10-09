#!/usr/bin/env bash
set -euo pipefail

source ~/venvs/dropwl_x86/bin/activate
export PYTHONPATH="$PWD"

# Seeds y rejillas "paper-like"
SEEDS=(20250925 20250926 20250927 20250928)
DEPTHS=(5 10 15 20)     # depth -> mlp_layers
WIDTHS=(2 10 20)        # width -> mlp_hidden
DIMS=(32 64)            # dimensión de salida del MLP por ejecución

mkdir -p results/paper_c4_capacity

for n in 8 16 24 32 40 44; do
  DATA="data_paper/c4_n${n}_p030_bal.npz"

  # ==== WL baseline (solo wl_*) ====
  OUT_WL="results/paper_c4_capacity/c4_n${n}/WL"
  mkdir -p "$OUT_WL"
  PYTHONPATH=. python scripts/exp_c4_compare.py \
    --data "$DATA" \
    --seeds "${SEEDS[@]}" \
    --wl_t 3 --wl_kmax "$n" \
    --outdir "$OUT_WL"

  # ==== dropWL-mean (solo dw_*) ====
  OUT_DW="results/paper_c4_capacity/c4_n${n}/dropWL-mean"
  mkdir -p "$OUT_DW"
  PYTHONPATH=. python scripts/exp_c4_compare.py \
    --data "$DATA" \
    --seeds "${SEEDS[@]}" \
    --dw_p 0.10 --dw_R 50 --dw_t 3 --dw_kmax "$n" \
    --standardize \
    --outdir "$OUT_DW"

  # ==== dropWL+MLP (solo mlp_*), barrido externo L×W×d ====
  for depth in "${DEPTHS[@]}"; do
    for width in "${WIDTHS[@]}"; do
      for d in "${DIMS[@]}"; do
        OUT_MLP="results/paper_c4_capacity/c4_n${n}/dropWL-MLP_L${depth}_W${width}_d${d}"
        mkdir -p "$OUT_MLP"
        PYTHONPATH=. python scripts/exp_c4_compare.py \
          --data "$DATA" \
          --seeds "${SEEDS[@]}" \
          --mlp_p 0.10 --mlp_R 50 --mlp_t 3 --mlp_kmax "$n" \
          --mlp_layers "$depth" --mlp_hidden "$width" --mlp_d "$d" --mlp_act relu \
          --standardize \
          --outdir "$OUT_MLP"
      done
    done
  done

  echo "[OK] n=${n} -> results/paper_c4_capacity/c4_n${n}"
done
