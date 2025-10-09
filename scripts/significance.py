#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
significance.py
---------------
Análisis de significancia sobre resultados con múltiples semillas:

Entradas:
  - Uno o varios directorios (--indir), cada uno con un 'results.csv'
    que contenga al menos: ['variant', 'acc_test', 'seed'].
  - Se asume que la carpeta es el output de compare_wl_drop_variants.py
    ejecutado con varias seeds.

Salidas:
  - Tablas CSV con estadísticas por variante (media, DE, IC bootstrap 95%).
  - Tablas CSV con deltas apareados (variant - WL) por seed, IC bootstrap 95%,
    y p-valor de prueba de permutación apareada (two-sided).
  - PDF con boxplots (accuracies y deltas) y encabezados del/los dataset(s).

Requisitos:
  - Python 3.8+
  - pandas, numpy, matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_results_csvs(indirs):
    rows = []
    for d in indirs:
        d = Path(d)
        csv = d / "results.csv"
        if not csv.exists():
            print(f"[WARN] No existe: {csv}")
            continue
        df = pd.read_csv(csv)
        # Propaga metadatos del dataset a las filas (nombre del experimento = nombre de carpeta)
        df["experiment"] = d.name
        rows.append(df)
    if not rows:
        raise RuntimeError("No se pudo leer ningún results.csv de los --indir proporcionados.")
    DF = pd.concat(rows, axis=0, ignore_index=True)
    # Chequeos mínimos
    for col in ["variant", "acc_test", "seed"]:
        if col not in DF.columns:
            raise RuntimeError(f"Falta la columna requerida '{col}' en los CSV.")
    return DF

def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    """
    IC bootstrap percentil (95% por defecto). No usa SciPy.
    """
    if rng is None:
        rng = np.random.RandomState(12345)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan, np.nan)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.randint(0, n, size=n)]
        boots.append(np.mean(sample))
    lo = np.percentile(boots, 100.0 * (alpha/2.0))
    hi = np.percentile(boots, 100.0 * (1.0 - alpha/2.0))
    return (float(lo), float(hi))

def paired_deltas_vs_wl(df):
    """
    Calcula deltas apareados (variant - WL) por 'seed' dentro de un 'experiment'.
    Retorna DataFrame con columnas: [experiment, seed, variant, delta]
    """
    out = []
    experiments = sorted(df["experiment"].unique())
    for exp in experiments:
        sub = df[df["experiment"] == exp]
        seeds = sorted(sub["seed"].unique())
        for s in seeds:
            sub_s = sub[sub["seed"] == s]
            wl_rows = sub_s[sub_s["variant"] == "WL"]
            if wl_rows.empty:
                continue
            wl_acc = float(wl_rows["acc_test"].mean())
            for v in sorted(sub_s["variant"].unique()):
                if v == "WL":
                    continue
                v_acc = float(sub_s[sub_s["variant"] == v]["acc_test"].mean())
                out.append({"experiment": exp, "seed": s, "variant": v, "delta": v_acc - wl_acc})
    if not out:
        return pd.DataFrame(columns=["experiment","seed","variant","delta"])
    return pd.DataFrame(out)

def permutation_test_paired_zero_mean(diffs, n_perm=100000, rng=None):
    """
    Prueba de permutación apareada (two-sided) para media de diferencias.
    Hipótesis nula: E[diff] = 0. Genera flips de signo aleatorios en las diferencias.

    Retorna: p-valor aproximado.
    """
    if rng is None:
        rng = np.random.RandomState(123)
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return np.nan
    obs = np.mean(diffs)
    k = diffs.size
    count = 0
    # Para k grande y n_perm grande, puede tardar; 100k suele ser razonable para decimales estables
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=k)
        m = np.mean(diffs * signs)
        if abs(m) >= abs(obs):
            count += 1
    p = (count + 1.0) / (n_perm + 1.0)  # corrección pequeña
    return float(p)

# ---------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------

def make_pdf_report(DF, tab_var, tab_pair, out_pdf):
    """
    Genera PDF con:
      - Encabezado (experimentos incluidos)
      - Boxplot accuracies por variante
      - Boxplot deltas (variant - WL) por experimento (si hay)
    """
    with PdfPages(out_pdf) as pdf:
        # Página 1: resumen y experiments
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        ax.axis('off')
        title = "Significance Report — dropWL vs WL"
        exps = ", ".join(sorted(DF["experiment"].unique()))
        txt = (
            f"{title}\n\n"
            f"Experimentos incluidos: {exps}\n"
            f"Filas totales: {len(DF)}  |  Variantes: {', '.join(sorted(DF['variant'].unique()))}\n"
            f"Fecha/hora de generación: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        ax.text(0.01, 0.8, txt, fontsize=11, va='top', ha='left')
        pdf.savefig(fig); plt.close(fig)

        # Página 2: boxplot accuracies por variante (todos los experimentos concatenados)
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        order = ["WL", "dropWL-mean", "dropWL+MLP"]
        data_box = [DF[DF["variant"]==v]["acc_test"].dropna().values for v in order if v in DF["variant"].unique()]
        labels = [v for v in order if v in DF["variant"].unique()]
        if data_box:
            ax.boxplot(data_box, labels=labels)
            ax.set_title("Accuracy (test) por variante")
            ax.set_ylabel("Accuracy (test)")
            ax.set_ylim(0.0, 1.02)
        else:
            ax.text(0.1, 0.5, "No hay datos para boxplot de accuracies", fontsize=11)
            ax.axis('off')
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 3+: para cada experimento, boxplot de deltas vs WL (si existen)
        deltas = paired_deltas_vs_wl(DF)
        if not deltas.empty:
            for exp in sorted(deltas["experiment"].unique()):
                sub = deltas[deltas["experiment"] == exp]
                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                variants = sorted([v for v in sub["variant"].unique() if v != "WL"])
                data_box = [sub[sub["variant"]==v]["delta"].values for v in variants]
                if data_box:
                    ax.boxplot(data_box, labels=variants)
                    ax.axhline(0.0, color='gray', linewidth=1, linestyle='--')
                    ax.set_title(f"Δ (variant − WL) por seed — {exp}")
                    ax.set_ylabel("Δ Accuracy (test)")
                else:
                    ax.text(0.1, 0.5, f"Sin deltas para {exp}", fontsize=11)
                    ax.axis('off')
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(6.0, 3.0))
            ax.text(0.1, 0.5, "No hay deltas apareados (variant − WL) para graficar.", fontsize=11)
            ax.axis('off')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, action="append", required=True,
                    help="Directorio con results.csv (puedes pasar múltiples --indir)")
    ap.add_argument("--outdir", type=str, default="results/significance_summary",
                    help="Directorio de salida para tablas y figuras")
    ap.add_argument("--n_boot", type=int, default=10000, help="n de remuestreos bootstrap")
    ap.add_argument("--n_perm", type=int, default=100000, help="n de permutaciones para prueba apareada")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabs = outdir / "tables"
    figs = outdir / "figures"
    ensure_dir(tabs); ensure_dir(figs)

    # Leer resultados
    DF = read_results_csvs(args.indir)

    # 1) Tabla por variante (media, DE, IC bootstrap)
    rows_var = []
    for (exp, var), sub in DF.groupby(["experiment","variant"]):
        vals = sub["acc_test"].astype(float).values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        lo, hi = bootstrap_ci(vals, n_boot=args.n_boot, alpha=0.05)
        rows_var.append({
            "experiment": exp,
            "variant": var,
            "mean_acc": mean,
            "std_acc": std,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n": int(len(vals))
        })
    tab_var = pd.DataFrame(rows_var).sort_values(["experiment","variant"])
    tab_var.to_csv(tabs / "acc_by_variant.csv", index=False)
    print("[table]", tabs / "acc_by_variant.csv")

    # 2) Tabla de deltas apareados (variant − WL) + IC bootstrap + p-permutación
    deltas = paired_deltas_vs_wl(DF)
    rows_pair = []
    if not deltas.empty:
        for (exp, var), sub in deltas.groupby(["experiment","variant"]):
            diffs = sub["delta"].astype(float).values
            mean = float(np.mean(diffs))
            std = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0
            lo, hi = bootstrap_ci(diffs, n_boot=args.n_boot, alpha=0.05)
            p = permutation_test_paired_zero_mean(diffs, n_perm=args.n_perm)
            rows_pair.append({
                "experiment": exp,
                "compare": f"{var} vs WL",
                "delta_mean": mean,
                "delta_std": std,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "p_perm_two_sided": p,
                "n_pairs": int(len(diffs))
            })
    tab_pair = pd.DataFrame(rows_pair).sort_values(["experiment","compare"]) if rows_pair else pd.DataFrame(
        columns=["experiment","compare","delta_mean","delta_std","ci95_lo","ci95_hi","p_perm_two_sided","n_pairs"]
    )
    tab_pair.to_csv(tabs / "paired_vs_WL.csv", index=False)
    print("[table]", tabs / "paired_vs_WL.csv")

    # 3) PDF con boxplots
    out_pdf = figs / "significance_report.pdf"
    make_pdf_report(DF, tab_var, tab_pair, out_pdf)
    print("[pdf]", out_pdf)

    print("\n[OK] Análisis de significancia completado.")

if __name__ == "__main__":
    main()
