#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_paper_c4_suite.py
Analiza resultados de run_paper_c4_suite.py y genera tablas + PDF por dataset.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    rng = np.random.RandomState(1234) if rng is None else rng
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (np.nan, np.nan)
    n = len(v)
    boots = [np.mean(v[rng.randint(0, n, size=n)]) for _ in range(n_boot)]
    lo = np.percentile(boots, 100*(alpha/2))
    hi = np.percentile(boots, 100*(1-alpha/2))
    return float(lo), float(hi)

def perm_test_delta(x, y, n_perm=20000, rng=None):
    """Prueba de permutación para media(x - y) vs 0."""
    rng = np.random.RandomState(42) if rng is None else rng
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    d0 = np.mean(x - y)
    n = len(x)
    cnt = 0
    for _ in range(n_perm):
        s = rng.randint(0, 2, size=n) * 2 - 1  # ±1
        d = np.mean(s * (x - y))
        if abs(d) >= abs(d0):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)

def paired_delta_vs_WL(df):
    out = []
    seeds = sorted(df['seed'].unique())
    for s in seeds:
        sub = df[df['seed']==s]
        wl = sub[sub['variant']=='WL']['acc_test'].mean()
        for v in sorted(sub['variant'].unique()):
            if v == 'WL': 
                continue
            val = sub[sub['variant']==v]['acc_test'].mean()
            out.append({'seed': s, 'variant': v, 'delta': float(val - wl)})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, required=True,
                    help="CSV maestro (results_master.csv) o uno por dataset")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    DF = pd.read_csv(args.master)
    if "dataset" not in DF.columns:
        DF["dataset"] = Path(args.master).stem

    outdir = Path(args.outdir) if args.outdir else Path(args.master).parent / "analysis"
    figs = outdir / "figures"; tabs = outdir / "tables"
    ensure_dir(figs); ensure_dir(tabs)

    # Por dataset
    for ds, sub in DF.groupby("dataset"):
        # Tablas básicas
        tab1 = sub.groupby("variant")["acc_test"].agg(["mean","std","count"]).reset_index()
        tab1.to_csv(tabs / f"{ds}_acc_by_variant.csv", index=False)

        # Paired Δ vs WL
        deltas = paired_delta_vs_WL(sub)
        if not deltas.empty:
            rows = []
            for v, g in deltas.groupby("variant"):
                lo, hi = bootstrap_ci(g["delta"].values, n_boot=10000, alpha=0.05)
                p = perm_test_delta(g["delta"].values, np.zeros_like(g["delta"].values), n_perm=20000)
                rows.append({"variant": v, "delta_mean": g["delta"].mean(), "ci_lo": lo, "ci_hi": hi, "p_perm": p, "n_pairs": len(g)})
            pd.DataFrame(rows).to_csv(tabs / f"{ds}_paired_vs_WL.csv", index=False)
        else:
            (tabs / f"{ds}_paired_vs_WL.csv").write_text("No paired deltas\n")

        # PDF (boxplots y barras)
        with PdfPages(figs / f"{ds}_report.pdf") as pdf:
            # Página 1: resumen
            fig, ax = plt.subplots(figsize=(8,3))
            ax.axis('off')
            ax.text(0.02, 0.88, f"Dataset: {ds}", fontsize=12, weight='bold')
            ax.text(0.02, 0.66, f"Filas: {len(sub)} | Variantes: {', '.join(sorted(sub['variant'].unique()))}", fontsize=10)
            ax.text(0.02, 0.48, time.strftime("%Y-%m-%d %H:%M:%S"), fontsize=10)
            pdf.savefig(fig); plt.close(fig)

            # Página 2: boxplot por variante
            fig, ax = plt.subplots(figsize=(5.5,3.5))
            order = ["WL", "dropWL-mean", "dropWL+MLP"]
            present = [v for v in order if v in sub["variant"].unique()]
            data = [sub[sub["variant"]==v]["acc_test"].values for v in present]
            ax.boxplot(data, labels=present)
            ax.set_title("Accuracy test por variante")
            ax.set_ylabel("Accuracy")
            ax.grid(alpha=0.3, linestyle="--")
            pdf.savefig(fig); plt.close(fig)

            # Página 3: barras (p,R) para dropWL-mean (si aplica)
            dw = sub[sub["variant"]=="dropWL-mean"].copy()
            if not dw.empty and {"p","R"}.issubset(dw.columns):
                grp = dw.groupby(["p","R"])["acc_test"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6,3.5))
                xticks = [f"p={row.p:.2f}\nR={int(row.R)}" for _, row in grp.iterrows()]
                ax.bar(range(len(grp)), grp["acc_test"].values)
                ax.set_xticks(range(len(grp))); ax.set_xticklabels(xticks, rotation=0)
                ax.set_title("dropWL-mean: media por (p,R)")
                ax.set_ylabel("Accuracy")
                ax.grid(alpha=0.3, linestyle="--")
                pdf.savefig(fig); plt.close(fig)

            # Página 4: barras (p,R,L,H,d,act) para dropWL+MLP (si aplica)
            mlp = sub[sub["variant"]=="dropWL+MLP"].copy()
            if not mlp.empty and {"p","R","layers","hidden","d","act"}.issubset(mlp.columns):
                grp = mlp.groupby(["p","R","layers","hidden","d","act"])["acc_test"].mean().reset_index()
                # top-12 configs por media
                grp = grp.sort_values("acc_test", ascending=False).head(12)
                fig, ax = plt.subplots(figsize=(7.5,4))
                xticks = [f"p={row.p:.2f},R={int(row.R)}\nL={int(row.layers)},H={int(row.hidden)},d={int(row.d)},{row.act}" for _, row in grp.iterrows()]
                ax.bar(range(len(grp)), grp["acc_test"].values)
                ax.set_xticks(range(len(grp))); ax.set_xticklabels(xticks, rotation=0, fontsize=8)
                ax.set_title("dropWL+MLP: Top-12 configs por accuracy media")
                ax.set_ylabel("Accuracy")
                ax.grid(alpha=0.3, linestyle="--")
                pdf.savefig(fig); plt.close(fig)

        print(f"[OK] {ds}: tablas en {tabs}, PDF en {figs}")

if __name__ == "__main__":
    main()
