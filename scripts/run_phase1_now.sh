#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS="20250925 20250926 20250927 20250928 20250929"
TEST_SIZE=0.30
T=3
KMAX=40
R=50
MLP_LAYERS=2
MLP_HIDDEN=128
MLP_D=64
MLP_ACT=relu
DEVICE=cpu
OUTROOT="results/fase1_simple_all"

# Ajusta el nivel de paralelismo aquí:
P=2   # (en Rosetta x86_64, 2 es prudente; puedes probar 3)

# Datasets existentes en tu árbol
DATASETS=(
  "C4_n8:data_paper/c4_n8_p030_bal.npz"
  "C4_n16:data_paper/c4_n16_p030_bal.npz"
  "C4_n24:data_paper/c4_n24_p030_bal.npz"
  "C4_n32:data_paper/c4_n32_p030_bal.npz"
  "C4_n40:data_paper/c4_n40_p030_bal.npz"
  "C4_n44:data_paper/c4_n44_p030_bal.npz"
  "LIMITS1:data_synth/limits/limits1_s20250925.npz"
  "LIMITS2:data_synth/limits/limits2_s20250925.npz"
  "SKIP32:data_synth/skip/skip_n32_s20250925.npz"
  "LCC32:data_synth/lcc/lcc_nodes_n32_s20250925.npz"
  "TRI32:data_synth/triangles/tri_nodes_n32_s20250925.npz"
)

mkdir -p "$OUTROOT"
> jobs_fase1.txt

make_job_line () {
  local name="$1"
  local npz="$2"
  local outdir="${OUTROOT}/${name}"
  mkdir -p "$outdir"
  if [[ -f "${outdir}/results.csv" ]]; then
    echo "# [skip] ${outdir}/results.csv ya existe" >> jobs_fase1.txt
    return
  fi
  cat >> jobs_fase1.txt <<CMD
PYTHONPATH=. python scripts/exp_simple_compare.py \
  --data "$npz" \
  --seeds $SEEDS \
  --test_size $TEST_SIZE \
  --t $T --kmax $KMAX --R $R \
  --run_wl --run_1drop_log --run_1drop_mlp \
  --mlp_layers $MLP_LAYERS --mlp_hidden $MLP_HIDDEN --mlp_d $MLP_D --mlp_act $MLP_ACT \
  --epochs 30 --lr 1e-3 --patience 8 \
  --device $DEVICE --outdir "$outdir" --standardize
CMD
}

# Construcción de jobs
for item in "${DATASETS[@]}"; do
  NAME="${item%%:*}"
  NPZ="${item#*:}"
  if [[ ! -f "$NPZ" ]]; then
    echo "[WARN] Dataset no encontrado: $NPZ" >&2
    continue
  fi
  make_job_line "$NAME" "$NPZ"
done

TOTAL=$(grep -vc '^[[:space:]]*#' jobs_fase1.txt || true)
echo "[INFO] Generados $TOTAL trabajos."

# ===== Ejecutor sin xargs (paralelo con semáforo) =====
if [[ "$TOTAL" -eq 0 ]]; then
  echo "[OK] Nada que ejecutar (todo existente)."
  exit 0
fi

running=0
while IFS= read -r line || [[ -n "$line" ]]; do
  # saltar comentarios y líneas vacías
  [[ -z "${line// }" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue

  # Espera si alcanzas el cupo de procesos
  while [[ $running -ge $P ]]; do
    # Espera a que termine alguno
    wait -n
    running=$((running-1))
  done

  echo "[RUN] $line"
  bash -lc "$line" &
  running=$((running+1))
done < jobs_fase1.txt

# Espera a que terminen todos
wait
echo "[OK] Fase 1 completada. Resultados en $OUTROOT"