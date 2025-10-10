# scripts/aggregate_phase1_simple.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Agrega resultados Fase 1 (WL, 1drop-LOG, 1drop-MLP)")
    ap.add_argument("--root", type=str, default="results/fase1_simple")
    ap.add_argument("--outcsv", type=str, default="results/fase1_simple/table_phase1_summary.csv")
    ap.add_argument("--outmd", type=str, default="results/fase1_simple/table_phase1_summary.md")
    ap.add_argument("--mastercsv", type=str, default="results/fase1_simple/all_runs_master.csv")
    args = ap.parse_args()

    root = Path(args.root)
    runs = []
    for f in root.glob("*/**/seed_*/results.csv"):
        try:
            df = pd.read_csv(f)
            # inferir dataset y modelo desde la ruta
            parts = f.parts
            ds = parts[parts.index(root.name)+1]
            model = parts[parts.index(root.name)+2]
            df["dataset"] = ds
            df["model"] = model
            runs.append(df)
        except Exception as e:
            print("[skip]", f, e)

    if not runs:
        print("[ERROR] No se encontraron runs. Revisa --root.")
        return

    DF = pd.concat(runs, ignore_index=True)
    DF.to_csv(args.mastercsv, index=False)
    print(f"[OK] maestro: {args.mastercsv} ({len(DF)} filas)")

    # columnas esperadas: acc_train, acc_test, seed, etc.
    # resumen media ± DE por dataset × modelo
    def agg(s):
        return pd.Series({
            "acc_train_mean": s["acc_train"].mean(),
            "acc_train_std":  s["acc_train"].std(ddof=0),
            "acc_test_mean":  s["acc_test"].mean(),
            "acc_test_std":   s["acc_test"].std(ddof=0),
            "n_runs": len(s)
        })

    T = DF.groupby(["dataset","model"]).apply(agg).reset_index()
    T.to_csv(args.outcsv, index=False)
    print(f"[OK] tabla CSV: {args.outcsv}")

    # versión Markdown (bonita para informe)
    def fmt_row(r):
        tr = f"{r['acc_train_mean']:.4f} ± {r['acc_train_std']:.4f}"
        te = f"{r['acc_test_mean']:.4f} ± {r['acc_test_std']:.4f}"
        return tr, te

    lines = ["| Dataset | Modelo | Train (media±DE) | Test (media±DE) | n |",
             "|:--|:--|--:|--:|--:|"]
    for _, r in T.sort_values(["dataset","model"]).iterrows():
        tr, te = fmt_row(r)
        lines.append(f"| {r['dataset']} | {r['model']} | {tr} | {te} | {int(r['n_runs'])} |")
    Path(args.outmd).write_text("\n".join(lines))
    print(f"[OK] tabla MD: {args.outmd}")

if __name__ == "__main__":
    main()
