# scripts/grid_tk_runner.py
"""
Barrido (t, kmax) manteniendo fijos p y R (definidos por --p y --R).
Ejecuta WL, dropWL-mean y dropWL+MLP por cada (t,kmax) y consolida a un CSV.
"""

import argparse, sys, subprocess
from pathlib import Path
from itertools import product

def run_cmd(cmd):
    print("\n[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Comando falló: {r.returncode}")
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--grid_t', nargs='+', type=int, required=True)
    ap.add_argument('--grid_kmax', nargs='+', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--R', type=int, required=True)
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
    base = Path(args.outdir_base); base.mkdir(parents=True, exist_ok=True)
    out_csv = base / "sweep_tk_results.csv"

    collected = []
    for t, kmax in product(args.grid_t, args.grid_kmax):
        tag = f"t{t}_k{kmax}"
        outdir = base / tag; outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "scripts/compare_wl_drop_variants.py",
            "--data", str(data),
            "--seeds", *[str(s) for s in args.seeds],
            "--run_wl", "--run_drop_mean", "--run_drop_mlp",
            "--wl_t", str(t), "--wl_kmax", str(kmax),
            "--dw_p", str(args.p), "--dw_R", str(args.R), "--dw_t", str(t), "--dw_kmax", str(kmax),
            "--mlp_p", str(args.p), "--mlp_R", str(args.R), "--mlp_t", str(t), "--mlp_kmax", str(kmax),
            "--mlp_layers", str(args.layers), "--mlp_hidden", str(args.hidden),
            "--mlp_d", str(args.d), "--mlp_act", args.act,
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--early_stop_patience", str(args.patience),
            "--standardize", "--time_each_step",
            "--outdir", str(outdir)
        ]
        run_cmd(cmd)

        import pandas as pd
        res = outdir / "results.csv"
        if res.exists():
            dfi = pd.read_csv(res)
            dfi['t'] = t; dfi['kmax'] = kmax
            collected.append(dfi)

    if collected:
        import pandas as pd
        DF = pd.concat(collected, axis=0, ignore_index=True)
        DF.to_csv(out_csv, index=False)
        print("\n== Resumen por (t,kmax,variant) ==")
        print(DF.groupby(['t','kmax','variant'])['acc_test'].agg(['mean','std','count']))
        print(f"\n[OK] Consolidado en: {out_csv}")
    else:
        print("[WARN] No se recolectaron resultados.")

if __name__ == "__main__":
    main()
