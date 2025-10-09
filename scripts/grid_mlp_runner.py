# scripts/grid_mlp_runner.py
"""
Barrido de arquitectura MLP manteniendo fijos p, R, t, kmax (los mejores estimados).
Explora: num_layers, hidden, d, activation. Ejecuta SOLO dropWL+MLP (y opcional WL/drop-mean como referencia rápida).
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
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--R', type=int, required=True)
    ap.add_argument('--t', type=int, required=True)
    ap.add_argument('--kmax', type=int, required=True)
    ap.add_argument('--grid_layers', nargs='+', type=int, required=True)
    ap.add_argument('--grid_hidden', nargs='+', type=int, required=True)
    ap.add_argument('--grid_d', nargs='+', type=int, required=True)
    ap.add_argument('--grid_act', nargs='+', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--outdir_base', type=str, required=True)
    ap.add_argument('--with_baselines', action='store_true',
                    help="Si se activa, corre también WL y dropWL-mean para referencia.")
    args = ap.parse_args()

    data = Path(args.data)
    base = Path(args.outdir_base); base.mkdir(parents=True, exist_ok=True)
    out_csv = base / "sweep_mlp_results.csv"

    collected = []
    for L, H, D, A in product(args.grid_layers, args.grid_hidden, args.grid_d, args.grid_act):
        tag = f"L{L}_H{H}_D{D}_A{A}"
        outdir = base / tag; outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "scripts/compare_wl_drop_variants.py",
            "--data", str(data),
            "--seeds", *[str(s) for s in args.seeds],
            "--mlp_p", str(args.p), "--mlp_R", str(args.R),
            "--mlp_t", str(args.t), "--mlp_kmax", str(args.kmax),
            "--mlp_layers", str(L), "--mlp_hidden", str(H),
            "--mlp_d", str(D), "--mlp_act", A,
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--early_stop_patience", str(args.patience),
            "--standardize", "--time_each_step",
            "--outdir", str(outdir),
            "--run_drop_mlp"
        ]
        if args.with_baselines:
            cmd += [
                "--run_wl", "--run_drop_mean",
                "--wl_t", str(args.t), "--wl_kmax", str(args.kmax),
                "--dw_p", str(args.p), "--dw_R", str(args.R),
                "--dw_t", str(args.t), "--dw_kmax", str(args.kmax),
            ]

        run_cmd(cmd)

        import pandas as pd
        res = outdir / "results.csv"
        if res.exists():
            dfi = pd.read_csv(res)
            dfi['layers'] = L; dfi['hidden'] = H; dfi['d'] = D; dfi['act'] = A
            collected.append(dfi)

    if collected:
        import pandas as pd
        DF = pd.concat(collected, axis=0, ignore_index=True)
        DF.to_csv(out_csv, index=False)
        print("\n== Resumen por (layers,hidden,d,act,variant) ==")
        print(DF.groupby(['layers','hidden','d','act','variant'])['acc_test'].agg(['mean','std','count']))
        print(f"\n[OK] Consolidado en: {out_csv}")
    else:
        print("[WARN] No se recolectaron resultados.")

if __name__ == "__main__":
    main()
