# scripts/analyze_ablations_triangles.py
"""
Fase 4.4 — Ablaciones y robustez (triángulos-hard)
Lee los resultados de 4.1/4.2/4.3 y genera tablas y figuras estilo paper.

Entradas esperadas:
- results/sweep_triangles_pR/sweep_results.csv
- results/sweep_triangles_tk/sweep_tk_results.csv
- results/sweep_triangles_mlp/sweep_mlp_results.csv

Salidas:
- results/ablations_triangles/
    tables/*.csv
    figures/*.png
    report_ablations.pdf

Requisitos:
- Python 3.8+ (o similar), pandas, matplotlib (no seaborn).
- Ejecutar en el mismo venv que generó los CSV (compatibilidad NumPy/pandas).
"""

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Utilidades
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def maybe_cols(df, cols):
    """Devuelve las columnas de 'cols' que existen en df, en orden."""
    return [c for c in cols if c in df.columns]

def agg_summary(df, by, val='acc_test'):
    g = df.groupby(by)[val].agg(['mean','std','count']).reset_index()
    return g

def pivot_delta_vs_wl(df, index_cols, variant_col='variant', acc_col='acc_test'):
    # calcula deltas (variant - WL) por grupos definidos por index_cols
    rows = []
    for _, sub in df.groupby(index_cols):
        # promedio WL en ese grupo
        wl = sub[sub[variant_col]=='WL'][acc_col].mean()
        if pd.isna(wl):
            continue
        for var in sub[variant_col].unique():
            if var == 'WL':
                continue
            m = sub[sub[variant_col]==var][acc_col].mean()
            rows.append({**{k:v for k,v in zip(index_cols, _ if isinstance(_, tuple) else (_,) )},
                         'variant': var, 'delta_vs_WL': m - wl})
    return pd.DataFrame(rows)

def plot_bars_mean_std(ax, table, x_col, y_mean='mean', y_std='std', title='', ylim=None, rotate=0):
    xs = table[x_col].astype(str).tolist()
    means = table[y_mean].tolist()
    stds = table[y_std].tolist()
    ax.bar(range(len(xs)), means)
    ax.errorbar(range(len(xs)), means, yerr=stds, fmt='none', capsize=3, linewidth=1)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=rotate)
    ax.set_ylabel('Accuracy (test)')
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_lines_group(ax, df, x, y, group, title='', xlabel=None, ylabel=None, x_is_int=False):
    groups = sorted(df[group].unique())
    for g in groups:
        sub = df[df[group]==g].sort_values(x)
        ax.plot(sub[x], sub[y], marker='o', label=f'{group}={g}')
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend()
    if x_is_int:
        ax.set_xticks(sorted(df[x].unique()))

def plot_scatter(ax, df, x, y, hue=None, title='', xlabel=None, ylabel=None):
    if hue and hue in df.columns:
        for hval in sorted(df[hue].unique()):
            sub = df[df[hue]==hval]
            ax.scatter(sub[x], sub[y], label=f'{hue}={hval}', s=30)
        ax.legend()
    else:
        ax.scatter(df[x], df[y], s=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

def save_table_csv(df, outdir, name):
    p = outdir / f'{name}.csv'
    df.to_csv(p, index=False)
    print(f'[table] {p}')

def save_fig(fig, outdir, name):
    p = outdir / f'{name}.png'
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'[fig] {p}')


# -----------------------------
# Main
# -----------------------------
def main():
    base_pr = Path('results/sweep_triangles_pR/sweep_results.csv')
    base_tk = Path('results/sweep_triangles_tk/sweep_tk_results.csv')
    base_mlp = Path('results/sweep_triangles_mlp/sweep_mlp_results.csv')

    out_base = Path('results/ablations_triangles')
    figs = out_base / 'figures'
    tabs = out_base / 'tables'
    ensure_dir(figs); ensure_dir(tabs)

    with PdfPages(out_base / 'report_ablations.pdf') as pdf:

        # =========================
        #  A) Analítica p-R
        # =========================
        if base_pr.exists():
            df = pd.read_csv(base_pr)
            keep = ['WL','dropWL-mean','dropWL+MLP']
            df = df[df['variant'].isin(keep)].copy()

            # tabla por (p,R,variant)
            tab_pr = agg_summary(df, by=['p','R','variant'])
            save_table_csv(tab_pr, tabs, 'A_pr_acc_by_variant')

            # barras por variante (promedio global)
            tab_var = agg_summary(df, by=['variant'])
            save_table_csv(tab_var, tabs, 'A_pr_acc_by_variant_global')

            # Figura A1: barras por variante (global)
            fig, ax = plt.subplots(figsize=(5,3.2))
            plot_bars_mean_std(ax, tab_var, x_col='variant', title='A1: Acc media por variante (p,R sweep)')
            save_fig(fig, figs, 'A1_pr_bar_by_variant'); pdf.savefig(fig)

            # Figura A2: acc vs R (por p) — una por variante
            for variant in keep:
                sub = df[df['variant']==variant]
                g = agg_summary(sub, by=['p','R'])
                fig, ax = plt.subplots(figsize=(5.5,3.2))
                plot_lines_group(ax, g, x='R', y='mean', group='p',
                                 title=f'A2: {variant} — Acc vs R por p', xlabel='R', ylabel='Acc test', x_is_int=True)
                save_fig(fig, figs, f'A2_pr_{variant}_acc_vs_R'); pdf.savefig(fig)

            # Figura A3: acc vs p (por R) — una por variante
            for variant in keep:
                sub = df[df['variant']==variant]
                g = agg_summary(sub, by=['R','p'])
                fig, ax = plt.subplots(figsize=(5.5,3.2))
                plot_lines_group(ax, g, x='p', y='mean', group='R',
                                 title=f'A3: {variant} — Acc vs p por R', xlabel='p', ylabel='Acc test', x_is_int=False)
                save_fig(fig, figs, f'A3_pr_{variant}_acc_vs_p'); pdf.savefig(fig)

            # Δ vs WL (tablas tipo “heatmap textual”)
            deltas = pivot_delta_vs_wl(df, index_cols=['p','R'])
            save_table_csv(deltas, tabs, 'A_pr_delta_vs_WL')

            # Si hay tiempos, acc vs costo
            time_cols = maybe_cols(df, ['time_repr','time_train','time_total'])
            if time_cols:
                # media de tiempos por (p,R,variant)
                tab_time = df.groupby(['p','R','variant'])[time_cols].mean().reset_index()
                save_table_csv(tab_time, tabs, 'A_pr_time_by_variant')

                # Figura A4: Acc vs time_total por variante
                for variant in keep:
                    sub = pd.merge(
                        tab_pr[tab_pr['variant']==variant][['p','R','variant','mean']],
                        tab_time[tab_time['variant']==variant][['p','R','variant','time_total']],
                        on=['p','R','variant'],
                        how='inner'
                    )
                    fig, ax = plt.subplots(figsize=(5,3.2))
                    plot_scatter(ax, sub, x='time_total', y='mean', title=f'A4: {variant} — Acc vs tiempo total', xlabel='Tiempo total (s)', ylabel='Acc test')
                    save_fig(fig, figs, f'A4_pr_{variant}_acc_vs_time'); pdf.savefig(fig)

        else:
            print('[WARN] No encontrado:', base_pr)

        # =========================
        #  B) Analítica t-kmax
        # =========================
        if base_tk.exists():
            df = pd.read_csv(base_tk)
            keep = ['WL','dropWL-mean','dropWL+MLP']
            df = df[df['variant'].isin(keep)].copy()

            tab_tk = agg_summary(df, by=['t','kmax','variant'])
            save_table_csv(tab_tk, tabs, 'B_tk_acc_by_variant')

            # Figura B1: barras por variante (promedio global)
            tab_var = agg_summary(df, by=['variant'])
            fig, ax = plt.subplots(figsize=(5,3.2))
            plot_bars_mean_std(ax, tab_var, x_col='variant', title='B1: Acc media por variante (t,kmax sweep)')
            save_table_csv(tab_var, tabs, 'B_tk_acc_by_variant_global')
            save_fig(fig, figs, 'B1_tk_bar_by_variant'); pdf.savefig(fig)

            # Figura B2: Acc vs t (por kmax) por variante
            for variant in keep:
                sub = df[df['variant']==variant]
                g = agg_summary(sub, by=['kmax','t'])
                fig, ax = plt.subplots(figsize=(5.5,3.2))
                plot_lines_group(ax, g, x='t', y='mean', group='kmax',
                                 title=f'B2: {variant} — Acc vs t por kmax', xlabel='t', ylabel='Acc test', x_is_int=True)
                save_fig(fig, figs, f'B2_tk_{variant}_acc_vs_t'); pdf.savefig(fig)

            # Figura B3: Acc vs kmax (por t) por variante
            for variant in keep:
                sub = df[df['variant']==variant]
                g = agg_summary(sub, by=['t','kmax'])
                fig, ax = plt.subplots(figsize=(5.5,3.2))
                plot_lines_group(ax, g, x='kmax', y='mean', group='t',
                                 title=f'B3: {variant} — Acc vs kmax por t', xlabel='kmax', ylabel='Acc test', x_is_int=True)
                save_fig(fig, figs, f'B3_tk_{variant}_acc_vs_kmax'); pdf.savefig(fig)

            deltas = pivot_delta_vs_wl(df, index_cols=['t','kmax'])
            save_table_csv(deltas, tabs, 'B_tk_delta_vs_WL')

        else:
            print('[WARN] No encontrado:', base_tk)

        # =========================
        #  C) Analítica MLP (arquitecturas)
        # =========================
        if base_mlp.exists():
            df = pd.read_csv(base_mlp)
            # Los barridos MLP pueden incluir también WL y drop-mean si --with_baselines estaba activo
            # Filtramos y tratamos cada uno.
            keep = ['WL','dropWL-mean','dropWL+MLP']
            df = df[df['variant'].isin(keep)].copy()

            # Resumen global por variante
            tab_var = agg_summary(df, by=['variant'])
            save_table_csv(tab_var, tabs, 'C_mlp_acc_by_variant_global')

            fig, ax = plt.subplots(figsize=(5,3.2))
            plot_bars_mean_std(ax, tab_var, x_col='variant', title='C1: Acc media por variante (MLP sweep)')
            save_fig(fig, figs, 'C1_mlp_bar_by_variant'); pdf.savefig(fig)

            # Solo MLP: tabla por arquitectura
            mlp = df[df['variant']=='dropWL+MLP'].copy()
            tab_arch = agg_summary(mlp, by=['layers','hidden','d','act'])
            save_table_csv(tab_arch, tabs, 'C_mlp_acc_by_architecture')

            # Figura C2: ranking top-10 arquitecturas
            top = tab_arch.sort_values('mean', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6,3.5))
            lbls = [f"L{r.layers}-H{r.hidden}-D{r.d}-{r.act}" for r in top.itertuples()]
            ax.bar(range(len(top)), top['mean'])
            ax.errorbar(range(len(top)), top['mean'], yerr=top['std'], fmt='none', capsize=3, linewidth=1)
            ax.set_xticks(range(len(top))); ax.set_xticklabels(lbls, rotation=45, ha='right')
            ax.set_ylabel('Accuracy (test)')
            ax.set_title('C2: TOP arquitecturas MLP (media ± DE)')
            save_fig(fig, figs, 'C2_mlp_top_arch'); pdf.savefig(fig)

            # Δ vs WL (si WL está presente en el sweep)
            deltas = pivot_delta_vs_wl(df, index_cols=['layers','hidden','d','act'])
            if not deltas.empty:
                save_table_csv(deltas, tabs, 'C_mlp_delta_vs_WL')

            # Si hay tiempo_total, acc vs costo para MLP
            time_cols = maybe_cols(df, ['time_repr','time_train','time_total'])
            if time_cols:
                sub = df[df['variant']=='dropWL+MLP']
                tab_time = sub.groupby(['layers','hidden','d','act'])[time_cols].mean().reset_index()
                save_table_csv(tab_time, tabs, 'C_mlp_time_by_arch')
                merged = pd.merge(tab_arch, tab_time, on=['layers','hidden','d','act'], how='inner')
                fig, ax = plt.subplots(figsize=(5,3.2))
                plot_scatter(ax, merged, x='time_total', y='mean', title='C3: MLP — Acc vs tiempo total',
                             xlabel='Tiempo total (s)', ylabel='Acc test')
                save_fig(fig, figs, 'C3_mlp_acc_vs_time'); pdf.savefig(fig)

        else:
            print('[WARN] No encontrado:', base_mlp)

    print(f"\n[OK] Reporte consolidado en {out_base/'report_ablations.pdf'}")
    print(f"[OK] Figuras en {figs}")
    print(f"[OK] Tablas en {tabs}")


if __name__ == '__main__':
    main()
