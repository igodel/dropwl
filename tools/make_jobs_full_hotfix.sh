#!/usr/bin/env bash

# CONFIG
NS="8 16 24 32 40 44"
SEEDS="20250925 20250926 20250927 20250928 20250929"
PGRID="0.1 0.3 0.5 0.7 0.9"
R=50
DATA_ROOT="data_paper"
OUTROOT="results/paper_full"
STANDARDIZE="--standardize"

# Limpia y crea jobs_full.txt
: > jobs_full.txt

for n in ${NS}; do
  DATA="${DATA_ROOT}/c4_n${n}_p030_bal.npz"
  KMAX="${n}"

  # WL
  echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --wl_t 3 --wl_kmax ${KMAX} ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/WL" >> jobs_full.txt

  # 1-DropWL-LOG (un solo borde eliminado por ejecución; clasificador logístico)
  echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --dw_t 3 --dw_kmax ${KMAX} --dw_R ${R} --onedrop ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/1drop-LOG" >> jobs_full.txt

  # 1-DropWL-MLP (un solo borde eliminado; MLPHead + LR)
  echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --mlp_t 3 --mlp_kmax ${KMAX} --mlp_R ${R} --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu --onedrop ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/1drop-MLP" >> jobs_full.txt

  # p-DropWL-LOG y p-DropWL-MLP (p en PGRID)
  for p in ${PGRID}; do
    echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --dw_p ${p} --dw_R ${R} --dw_t 3 --dw_kmax ${KMAX} ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/p${p}-LOG" >> jobs_full.txt
    echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data ${DATA} --seeds ${SEEDS} --mlp_p ${p} --mlp_R ${R} --mlp_t 3 --mlp_kmax ${KMAX} --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu ${STANDARDIZE} --outdir ${OUTROOT}/c4_n${n}/p${p}-MLP" >> jobs_full.txt
  done
done

# Resumen
echo "[INFO] jobs_full.txt generado con $(wc -l < jobs_full.txt) trabajos."
