#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis para Fase 7 – Ablaciones formales:
  - Placebo p=0
  - Orden: mean->MLP vs MLP->mean
  - Estandarización: ON vs OFF

Lee results/ablations_phase7/<dataset>/results.csv y produce:
  - Tablas CSV con medias, DE, Δ vs WL (apareadas por seed) e ICs bootstrap.
  - PDF con figuras resumen (boxplots y barras por condición).
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
    if rng is None:
        rng = np.random.RandomState(12345)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan, np.nan)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = values[rng.randint(0, n, size=n)]
        boots.append(np.mean(sample))
    lo = np.percentile(boots, 100*(alpha/2))
    hi = np.percentile(boots, 100*(1-alpha/2))
    return float(lo), float(hi)

def paired_deltas_vs_WL(df):
    out = []
    seeds = sorted(df['seed'].unique())
    for s in seeds:
        sub = df[df['seed']==s]
        wl_rows = sub[sub['variant']=='WL']
        if wl_rows.empty: 
            continue
        wl = float(wl_rows['acc_test'].mean())
        for v in sorted(sub['variant'].unique()):
            if v == 'WL': 
                continue
            val = float(sub[sub['variant']==v]['acc_test'].mean())
            out.append({'seed': s, 'variant': v, 'delta': val - wl})
    return pd.DataFrame(out)

def plot_box(ax, data, labels, title, ylabel):
    if len(data) == 0:
        ax.text(0.1, 0.5, "Sin datos", fontsize=11); ax.axis('off'); return
    ax.boxplot(data, labels=labels)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True,
                    help="Directorio con results.csv (output de run_ablations_phase7.py)")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir / "analysis"
    tabs = outdir / "tables"; figs = outdir / "figures"
    ensure_dir(tabs); ensure_dir(figs)

    csv = indir / "results.csv"
    DF = pd.read_csv(csv)

    # -------- Tablas generales --------
    # Media±DE por (variant, exp, note)
    tab1 = DF.groupby(['variant','exp','note'])['acc_test'].agg(['mean','std','count']).reset_index()
    tab1.to_csv(tabs/'means_by_variant_exp_note.csv', index=False)

    # Δ vs WL apareado (por seed)
    deltas = paired_deltas_vs_WL(DF)
    if not deltas.empty:
        tab2 = deltas.groupby('variant')['delta'].agg(['mean','std','count']).reset_index()
        # ICs bootstrap por variante
        rows_ci = []
        for v, sub in deltas.groupby('variant'):
            lo, hi = bootstrap_ci(sub['delta'].values, n_boot=args.n_boot, alpha=0.05)
            rows_ci.append({'variant': v, 'ci95_lo': lo, 'ci95_hi': hi})
        tab2 = tab2.merge(pd.DataFrame(rows_ci), on='variant', how='left')
        tab2.to_csv(tabs/'paired_delta_vs_WL.csv', index=False)
    else:
        (tabs/'paired_delta_vs_WL.csv').write_text("no deltas\n")

    # -------- Figuras (PDF) --------
    with PdfPages(figs/'ablations_report.pdf') as pdf:
        # Página 1: resumen
        fig, ax = plt.subplots(figsize=(8,3))
        ax.axis('off')
        ax.text(0.02, 0.9, "Fase 7 – Ablaciones formales", fontsize=12, weight='bold')
        ax.text(0.02, 0.7, f"Dataset: {indir.name}", fontsize=11)
        ax.text(0.02, 0.55, f"Filas: {len(DF)} | Variantes: {', '.join(sorted(DF['variant'].unique()))}", fontsize=10)
        ax.text(0.02, 0.35, f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # A) Placebo p=0
        placebos = DF[DF['exp'].str.contains('placebo_p0')]
        fig, ax = plt.subplots(figsize=(6,3))
        if not placebos.empty:
            for std_flag in [False, True]:
                sub = placebos[placebos['note'].str.contains(f"std={std_flag}")]
                labs = []
                data = []
                for order in ['mean->MLP','MLP->mean']:
                    arr = sub[(sub['variant']=='dropWL+MLP') & (sub['order']==order)]['acc_test'].values
                    if arr.size:
                        labs.append(f"MLP p=0 ({order}, std={std_flag})")
                        data.append(arr)
                arr_mean = placebos[(placebos['variant']=='dropWL-mean') & (placebos['p']==0.0) & (placebos['standardize']==std_flag)]['acc_test'].values
                if arr_mean.size:
                    labs.append(f"mean p=0 (std={std_flag})")
                    data.append(arr_mean)
                wl_arr = DF[(DF['variant']=='WL') & (DF['standardize']==std_flag)]['acc_test'].values
                labs.append(f"WL (std={std_flag})"); data.append(wl_arr)
                plot_box(ax, data, labs, f"A) Placebo p=0 (std={std_flag})", "Accuracy test")
        else:
            ax.text(0.1, 0.5, "No hay datos de placebo p=0", fontsize=11); ax.axis('off')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # B) Orden p>0
        orderp = DF[DF['exp']=='order_p>0']
        for std_flag in [False, True]:
            fig, ax = plt.subplots(figsize=(6,3))
            sub = orderp[orderp['standardize']==std_flag]
            labs, data = [], []
            for order in ['mean->MLP','MLP->mean']:
                arr = sub[(sub['variant']=='dropWL+MLP') & (sub['order']==order)]['acc_test'].values
                if arr.size:
                    labs.append(f"{order} (std={std_flag})"); data.append(arr)
            wl_arr = DF[(DF['variant']=='WL') & (DF['standardize']==std_flag)]['acc_test'].values
            labs.append(f"WL (std={std_flag})"); data.append(wl_arr)
            plot_box(ax, data, labs, f"B) Orden (p>0) (std={std_flag})", "Accuracy test")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # C) Estandarización (dropWL-mean, p>0)
        stdm = DF[DF['exp']=='std_effect_mean']
        fig, ax = plt.subplots(figsize=(6,3))
        labs, data = [], []
        arr_no = stdm[stdm['standardize']==False]['acc_test'].values
        arr_si = stdm[stdm['standardize']==True]['acc_test'].values
        if arr_no.size: labs.append("dropWL-mean std=OFF"); data.append(arr_no)
        if arr_si.size: labs.append("dropWL-mean std=ON"); data.append(arr_si)
        wl_all = DF[DF['variant']=='WL']['acc_test'].values
        labs.append("WL (todas std)"); data.append(wl_all)
        plot_box(ax, data, labs, "C) Estandarización (dropWL-mean, p>0)", "Accuracy test")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print("[OK] Tablas en:", tabs)
    print("[OK] PDF en   :", figs/'ablations_report.pdf')

if __name__ == "__main__":
    main()
