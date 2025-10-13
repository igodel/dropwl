#!/usr/bin/env bash
set -euo pipefail

NS="8 16 24 32 40 44"           # tamaños para C4
SEEDS="20250925 20250926 20250927 20250928 20250929"   # S=5
PGRID="0.1 0.3 0.5 0.7 0.9"     # p-grid
R=50
DATA_ROOT="data_paper"
OUTROOT="results/paper_full"
STANDARDIZE="--standardize"

: > jobs_full.txt

# helper kmax (para C4 usamos kmax=n)
kmax_for() { echo "$1"; }

emit(){ echo "$@" >> jobs_full.txt; }

# --- 1) C4 (detección de 4-ciclos) ---
for n in ${NS}; do
  DATA="${DATA_ROOT}/c4_n${n}_p030_bal.npz"
  KMAX="$(kmax_for ${n})"

  # WL
  emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --wl_t 3 --wl_kmax ${KMAX} ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/WL"

  # 1-DropWL-LOG (one-edge-drop; clasificador logístico)
  emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --dw_t 3 --dw_kmax ${KMAX} --dw_R ${R} --onedrop ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/1drop-LOG"

  # 1-DropWL-MLP (one-edge-drop; MLPHead + LR)
  emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --mlp_t 3 --mlp_kmax ${KMAX} --mlp_R ${R} --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu --onedrop ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/1drop-MLP"

  # p-DropWL-LOG y p-DropWL-MLP: p ∈ {0.1,0.3,0.5,0.7,0.9}
  for p in ${PGRID}; do
    emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --dw_p ${p} --dw_R ${R} --dw_t 3 --dw_kmax ${KMAX} ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/p${p}-LOG"
    emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --mlp_p ${p} --mlp_R ${R} --mlp_t 3 --mlp_kmax ${KMAX} --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/p${p}-MLP"
  done
done

# --- 2) Placeholders para los otros 5 datasets (cuando estén listos en data_paper/*.npz) ---
# for name in limits1 limits2 lcc triangles skip; do
#   DATA="${DATA_ROOT}/${name}_n1000_bal.npz"; KMAX=20
#   emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --wl_t 3 --wl_kmax ${KMAX} ${STANDARDIZE} --outdir ${OUTROOT}/${name}/WL"
#   emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --dw_t 3 --dw_kmax ${KMAX} --dw_R ${R} --onedrop ${STANDARDIZE} --outdir ${OUTROOT}/${name}/1drop-LOG"
#   emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --mlp_t 3 --mlp_kmax ${KMAX} --mlp_R ${R} --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu --onedrop ${STANDARDIZE} --outdir ${OUTROOT}/${name}/1drop-MLP"
#   for p in ${PGRID}; do
#     emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --dw_p ${p} --dw_R ${R} --dw_t 3 --dw_kmax ${KMAX} ${STANDARDIZE} --outdir ${OUTROOT}/${name}/p${p}-LOG"
#     emit "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --mlp_p ${p} --mlp_R ${R} --mlp_t 3 --mlp_kmax ${KMAX} --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu ${STANDARDIZE} --outdir ${OUTROOT}/${name}/p${p}-MLP"
#   done
# done
