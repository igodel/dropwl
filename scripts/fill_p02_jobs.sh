set -euo pipefail


NS="8 16 24 32 40 44"
SEEDS="20250925 20250926 20250927 20250928"
DATA_ROOT="data_paper"
OUTROOT="results/paper_c4_full"

for n in $NS; do
  DATA="$DATA_ROOT/c4_n${n}_p030_bal.npz"

  OUT_MEAN="$OUTROOT/c4_n${n}/dropWL-mean"
  if [ ! -f "$OUT_MEAN/results.csv" ]; then
    echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data $DATA --seeds $SEEDS --dw_p 0.2 --dw_R 50 --dw_t 3 --dw_kmax $n --standardize --outdir $OUT_MEAN"
  fi

  OUT_MLP="$OUTROOT/c4_n${n}/dropWL-MLP"
  if [ ! -f "$OUT_MLP/results.csv" ]; then
    echo "PYTHONPATH=. python scripts/exp_c4_compare.py --data $DATA --seeds $SEEDS --mlp_p 0.2 --mlp_R 50 --mlp_t 3 --mlp_kmax $n --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu --standardize --outdir $OUT_MLP"
  fi
done
