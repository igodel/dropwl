# scripts/check_2c4_vs_c8.py
# -*- coding: utf-8 -*-
"""
Revisión del experimento 2C4 vs C8 (WL vs dropWL-mean vs dropWL+MLP)

Funciones:
- Si existe results.csv en --results_dir: carga, resume y grafica.
- Si NO existe results.csv y se pasa --run_if_missing:
    * Si no existe dataset --data, aborta (primero genera el .npz).
    * Ejecuta scripts/compare_wl_drop_variants.py con parámetros por defecto.
    * Luego carga resultados y genera resumen + figura.

Salidas en --results_dir:
- results.csv (si se corrió)
- summary.csv (media, std, count por variante)
- paired_vs_WL.csv (delta por semilla contra WL)
- box_acc.png (boxplot accuracy de test)
"""

import argparse
from pathlib import Path
import subprocess, shlex, sys
import pandas as pd
import numpy as np

def run_compare_if_needed(results_dir: Path, data_path: Path, seeds):
    """
    Lanza el experimento si no existe results.csv y el usuario lo pidió.
    """
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        return

    # Verificaciones
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {data_path}\n"
            f"Primero genera el .npz de 2C4 vs C8 antes de usar --run_if_missing."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"PYTHONPATH=. python scripts/compare_wl_drop_variants.py "
        f"--data {shlex.quote(str(data_path))} "
        f"--seeds {' '.join(str(s) for s in seeds)} "
        f"--run_wl --run_drop_mean --run_drop_mlp "
        f"--wl_t 3 --wl_kmax 20 "
        f"--dw_p 0.1 --dw_R 50 --dw_t 3 --dw_kmax 20 "
        f"--mlp_p 0.1 --mlp_R 50 --mlp_t 3 --mlp_kmax 20 "
        f"--mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu "
        f"--epochs 30 --lr 1e-3 --early_stop_patience 8 "
        f"--standardize "
        f"--outdir {shlex.quote(str(results_dir))} "
        f"--learning_curves"
    )
    print("[run] ", cmd)
    # Ejecutamos en shell simple para respetar PYTHONPATH=.
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError("Fallo la ejecución del comparador.")


def summarize_results(results_dir: Path):
    """
    Lee results.csv y genera:
      - summary.csv (media, std, count por variante)
      - paired_vs_WL.csv (delta por semilla vs WL)
      - box_acc.png (boxplot)
    """
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"No existe {results_csv}.")
    df = pd.read_csv(results_csv)

    # Resumen por variante
    summ = df.groupby("variant")["acc_test"].agg(["mean","std","count"]).reset_index()
    summ.to_csv(results_dir / "summary.csv", index=False)

    # Paired vs WL (por semilla): Δ = acc(variant) - acc(WL) con misma seed
    pairs = []
    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed]
        wl = sub[sub["variant"] == "WL"]["acc_test"]
        if len(wl) == 0:
            continue
        wl_val = float(wl.mean())
        for v in sub["variant"].unique():
            if v == "WL":
                continue
            val = float(sub[sub["variant"] == v]["acc_test"].mean())
            pairs.append({"seed": seed, "variant": v, "delta_vs_WL": val - wl_val})
    if pairs:
        paired = pd.DataFrame(pairs)
        paired.to_csv(results_dir / "paired_vs_WL.csv", index=False)

    # Figura: boxplot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.0, 4.0))
        order = ["WL", "dropWL-mean", "dropWL+MLP"]
        data = [df[df["variant"] == v]["acc_test"].values for v in order if v in df["variant"].unique()]
        labels = [v for v in order if v in df["variant"].unique()]
        plt.boxplot(data, labels=labels)
        plt.ylabel("Accuracy (test)")
        plt.title("2C4 vs C8: WL vs dropWL")
        plt.tight_layout()
        plt.savefig(results_dir / "box_acc.png", dpi=200)
        plt.close()
    except Exception as e:
        print("[WARN] No se pudo generar boxplot:", repr(e))

    # Imprime al final
    print("\n== Resumen por variante ==")
    print(summ)
    if pairs:
        print("\n== Deltas vs WL (por semilla) ==")
        print(paired.groupby("variant")["delta_vs_WL"].agg(["mean","std","count"]).reset_index())


def main():
    ap = argparse.ArgumentParser(description="Chequeo 2C4 vs C8 (WL vs dropWL)")
    ap.add_argument("--results_dir", type=str, default="results/cycles_2c4_vs_c8",
                    help="Carpeta donde está (o se guardará) results.csv")
    ap.add_argument("--data", type=str, default="data/cycles_2c4_vs_c8.npz",
                    help="Ruta al dataset .npz (para --run_if_missing)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20250925, 20250926, 20250927, 20250928],
                    help="Semillas a usar si se corre el experimento")
    ap.add_argument("--run_if_missing", action="store_true",
                    help="Si no existe results.csv, corre el experimento")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    data_path = Path(args.data)

    if not (results_dir / "results.csv").exists():
        if args.run_if_missing:
            print("[info] results.csv no encontrado. Ejecutando experimento...")
            run_compare_if_needed(results_dir, data_path, args.seeds)
        else:
            raise FileNotFoundError(
                f"No existe {results_dir/'results.csv'}.\n"
                f"Usa --run_if_missing para ejecutar el experimento automáticamente."
            )

    summarize_results(results_dir)


if __name__ == "__main__":
    main()
