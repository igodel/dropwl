# scripts/analyze_scaling_triangles.py
"""
Fase 5 — Escalamiento sintético (triángulos-hard)
Consolida resultados de n=20, n=40 (y opcional n=80) y produce tablas y figuras:
- Accuracy por variante vs n
- Δ (dropWL - WL) vs n
- Tiempo (representación / entrenamiento / total) vs n
- Curvas de aprendizaje (si existen archivos por variante en cada outdir)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_results_with_n(path: Path, n_value: int):
    csv = path / "results.csv"
    if not csv.exists():
        print(f"[WARN] No encontrado: {csv}")
        return None
    df = pd.read_csv(csv)
    df['n'] = n_value
    return df

def bars_by_variant(ax, table, title=''):
    xs = table['variant'].tolist()
    means = table['mean'].tolist()
    stds = table['std'].tolist()
    ax.bar(range(len(xs)), means)
    ax.errorbar(range(len(xs)), means, yerr=stds, fmt='none', capsize=3, linewidth=1)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=0)
    ax.set_ylabel('Accuracy (test)')
    ax.set_title(title)

def lines_by_n(ax, df, ycol, title='', ylabel=''):
    # df: columns [n, variant, ycol]
    variants = ['WL','dropWL-mean','dropWL+MLP']
    for v in variants:
        sub = df[df['variant']==v].sort_values('n')
        if sub.empty: 
            continue
        ax.plot(sub['n'], sub[ycol], marker='o', label=v)
    ax.set_title(title); ax.set_xlabel('n'); ax.set_ylabel(ylabel or ycol)
    ax.legend()

def compute_deltas_vs_wl(df):
    rows = []
    for (n, seed), sub in df.groupby(['n','seed']):
        wl = sub[sub['variant']=='WL']['acc_test'].mean()
        for v in sub['variant'].unique():
            if v == 'WL': 
                continue
            val = sub[sub['variant']==v]['acc_test'].mean()
            rows.append({'n':n, 'seed':seed, 'variant':v, 'delta_vs_WL': val - wl})
    return pd.DataFrame(rows)

def main():
    # Ajusta si cambiaste carpetas
    inputs = [
        ('results/triangles_hard_n20_compare', 20),
        ('results/triangles_hard_n40_compare', 40),
        ('results/triangles_hard_n80_compare', 80),
    ]
    frames = []
    for d, n in inputs:
        df = load_results_with_n(Path(d), n)
        if df is not None:
            frames.append(df)
    if not frames:
        print("[ERROR] No se encontraron resultados.")
        return
    DF = pd.concat(frames, axis=0, ignore_index=True)

    outdir = Path('results/scaling_triangles')
    figs = outdir/'figures'; tabs = outdir/'tables'
    ensure(figs); ensure(tabs)

    # Tabla: accuracy media por (n, variant)
    tab_nv = DF.groupby(['n','variant'])['acc_test'].agg(['mean','std','count']).reset_index()
    tab_nv.to_csv(tabs/'acc_by_n_variant.csv', index=False)
    print('[table]', tabs/'acc_by_n_variant.csv')

    # Figura 1: barras por variante (para cada n)
    with PdfPages(outdir/'report_scaling.pdf') as pdf:
        for n in sorted(tab_nv['n'].unique()):
            sub = tab_nv[tab_nv['n']==n]
            fig, ax = plt.subplots(figsize=(5,3.2))
            bars_by_variant(ax, sub, title=f'Acc media por variante — n={n}')
            fig.tight_layout(); fig.savefig(figs/f'F1_bar_by_variant_n{n}.png', dpi=150); pdf.savefig(fig); plt.close(fig)

        # Figura 2: accuracy vs n (líneas por variante)
        fig, ax = plt.subplots(figsize=(5.5,3.2))
        lines_by_n(ax, tab_nv.rename(columns={'mean':'acc_mean'}), ycol='acc_mean',
                   title='Accuracy (test) vs n', ylabel='Accuracy (media)')
        fig.tight_layout(); fig.savefig(figs/'F2_acc_vs_n.png', dpi=150); pdf.savefig(fig); plt.close(fig)

        # Tabla y figura: Δ vs WL por n
        gaps = compute_deltas_vs_wl(DF)
        tab_gap = gaps.groupby(['n','variant'])['delta_vs_WL'].agg(['mean','std','count']).reset_index()
        tab_gap.to_csv(tabs/'delta_vs_WL_by_n.csv', index=False)
        print('[table]', tabs/'delta_vs_WL_by_n.csv')

        fig, ax = plt.subplots(figsize=(5.5,3.2))
        for v in sorted(tab_gap['variant'].unique()):
            sub = tab_gap[tab_gap['variant']==v].sort_values('n')
            ax.plot(sub['n'], sub['mean'], marker='o', label=v)
            ax.fill_between(sub['n'], sub['mean']-sub['std'], sub['mean']+sub['std'], alpha=0.2)
        ax.set_title('Δ (variant − WL) vs n'); ax.set_xlabel('n'); ax.set_ylabel('Δ accuracy (media ± DE)')
        ax.legend()
        fig.tight_layout(); fig.savefig(figs/'F3_delta_vs_wl.png', dpi=150); pdf.savefig(fig); plt.close(fig)

        # Costos (si existen columnas de tiempo)
        time_cols = [c for c in ['time_repr','time_train','time_total'] if c in DF.columns]
        if time_cols:
            tab_time = DF.groupby(['n','variant'])[time_cols].mean().reset_index()
            tab_time.to_csv(tabs/'time_by_n_variant.csv', index=False)
            print('[table]', tabs/'time_by_n_variant.csv')

            # Figura 4: tiempo total vs n
            if 'time_total' in time_cols:
                fig, ax = plt.subplots(figsize=(5.5,3.2))
                lines_by_n(ax, tab_time.rename(columns={'time_total':'t_total_mean'}), ycol='t_total_mean',
                           title='Tiempo total vs n', ylabel='Tiempo total (s)')
                fig.tight_layout(); fig.savefig(figs/'F4_time_total_vs_n.png', dpi=150); pdf.savefig(fig); plt.close(fig)

    print('[OK] Reporte:', outdir/'report_scaling.pdf')
    print('[OK] Figuras en:', figs)
    print('[OK] Tablas  en:', tabs)

if __name__ == '__main__':
    main()
