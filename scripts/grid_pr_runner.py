# scripts/grid_pr_runner.py
"""
Orquestador de barrido (p, R) para triángulos (o cualquier dataset .npz).
Llama a compare_wl_drop_variants.py por cada par (p,R), ejecutando:
  - WL baseline
  - dropWL-mean (con esos p,R)
  - dropWL+MLP   (con esos p,R)
y consolida todos los results.csv en un solo CSV global.

Requisitos:
- PYTHONPATH=. para que compare_wl_drop_variants.py importe el repo.
- Mismo venv para crear/leer datasets .npz.
- PyTorch instalado si se activa MLP.

Uso:
  PYTHONPATH=. python scripts/grid_pr_runner.py \
    --data data/triangles_hard_n20.npz \
    --grid_p 0.05 0.1 0.2 \
    --grid_R 20 50 100 \
    --t 3 --kmax 20 \
    --layers 2 --hidden 128 --d 64 --act relu \
    --epochs 15 --lr 1e-3 --patience 6 \
    --seeds 20250925 20250926 20250927 \
    --outdir_base results/sweep_triangles_pR
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from itertools import product

def run_cmd(cmd):
    print("\n[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Comando falló con código {r.returncode}")
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--grid_p', nargs='+', type=float, required=True)
    ap.add_argument('--grid_R', nargs='+', type=int, required=True)
    ap.add_argument('--t', type=int, default=3)
    ap.add_argument('--kmax', type=int, default=20)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--act', type=str, default='relu', choices=['relu','tanh'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--outdir_base', type=str, required=True)
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        print(f"[ERROR] No existe dataset: {data}", file=sys.stderr)
        sys.exit(1)

    base = Path(args.outdir_base)
    base.mkdir(parents=True, exist_ok=True)

    # guardaremos aqui un CSV consolidado
    out_csv = base / "sweep_results.csv"
    collected_rows = []

    # barrido (p,R)
    for p, R in product(args.grid_p, args.grid_R):
        tag = f"p{p}_R{R}"
        outdir = base / tag
        outdir.mkdir(parents=True, exist_ok=True)

        # Ejecutar las TRES variantes con compare_wl_drop_variants.py
        cmd = [
            sys.executable, "scripts/compare_wl_drop_variants.py",
            "--data", str(data),
            "--seeds", *[str(s) for s in args.seeds],
            "--run_wl", "--run_drop_mean", "--run_drop_mlp",
            "--wl_t", str(args.t), "--wl_kmax", str(args.kmax),
            "--dw_p", str(p), "--dw_R", str(R), "--dw_t", str(args.t), "--dw_kmax", str(args.kmax),
            "--mlp_p", str(p), "--mlp_R", str(R), "--mlp_t", str(args.t), "--mlp_kmax", str(args.kmax),
            "--mlp_layers", str(args.layers), "--mlp_hidden", str(args.hidden),
            "--mlp_d", str(args.d), "--mlp_act", args.act,
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--early_stop_patience", str(args.patience),
            "--standardize", "--time_each_step",
            "--outdir", str(outdir)
        ]
        run_cmd(cmd)

        # Leer el results.csv de ese outdir y anexar (añadimos p,R para consolidar)
        res = outdir / "results.csv"
        if not res.exists():
            print(f"[WARN] No se encontró {res}, se omite.")
            continue

        import pandas as pd
        dfi = pd.read_csv(res)
        dfi['p'] = p
        dfi['R'] = R
        collected_rows.append(dfi)

    if not collected_rows:
        print("[WARN] No se recolectó ningún results.csv")
        sys.exit(0)

    import pandas as pd
    DF = pd.concat(collected_rows, axis=0, ignore_index=True)
    DF.to_csv(out_csv, index=False)

    # pequeño resumen en consola
    print("\n== Resumen por (p,R,variant) ==")
    print(DF.groupby(['p','R','variant'])['acc_test'].agg(['mean','std','count']))

    print(f"\n[OK] Sweep consolidado en: {out_csv}")

if __name__ == "__main__":
    main()
