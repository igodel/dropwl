#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Informe extendido (paper-like) para WL vs dropWL-mean vs dropWL+MLP
sobre el suite C4 (4-cycles), con múltiples vistas:

1) Curvas mean±std (train y test) vs n por p
2) Barras agrupadas (test) por n (por p)
3) Scatter acc_train vs acc_test (por p)
4) Heatmaps de sensibilidad (si hay columnas disponibles):
   - dropWL-mean: acc_test vs (R, t)
   - dropWL+MLP:  acc_test vs (layers, d)
5) Δ (ganancia) vs WL por n (para dropWL-mean y dropWL+MLP)
6) Boxplots de acc_test por variante y n (por p)
7) CSVs de resumen y deltas

Requisitos: numpy, pandas, matplotlib.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import itertools
import warnings

# ----------------------------
# Utilidades de carga / limpieza
# ----------------------------

def load_all_results(root: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(root.glob('c4_n*/**/results.csv')):
        try:
            df = pd.read_csv(csv)
            # Inferir n desde la ruta c4_nXX
            n = int(csv.parts[csv.parts.index('paper_c4_full')+1].split('c4_n')[1])
            df['n'] = n
            # Nombre de carpeta variante por si acaso
            df['variant_dir'] = csv.parent.name
            rows.append(df)
        except Exception as e:
            print("[WARN] no pude leer:", csv, repr(e))
    if not rows:
        raise RuntimeError(f"No encontré results.csv bajo {root}")
    DF = pd.concat(rows, ignore_index=True)
    return DF

def reconstruct_p(DF: pd.DataFrame) -> pd.DataFrame:
    DF = DF.copy()
    if 'p' not in DF.columns:
        DF['p'] = np.nan
    if 'variant' in DF.columns:
        if 'dw_p' in DF.columns:
            m = DF['variant'].eq('dropWL-mean')
            DF.loc[m, 'p'] = DF.loc[m, 'dw_p']
        if 'mlp_p' in DF.columns:
            m = DF['variant'].eq('dropWL+MLP')
            DF.loc[m, 'p'] = DF.loc[m, 'mlp_p']
        # WL queda NaN y se mantiene como referencia
    return DF

def ensure_columns(DF: pd.DataFrame) -> pd.DataFrame:
    # Rellenar algunas columnas esperadas si faltan
    DF = DF.copy()
    for col in ['acc_train','acc_test','variant','n']:
        if col not in DF.columns:
            raise ValueError(f"Falta columna obligatoria '{col}' en DF.")
    return DF

# ----------------------------
# Resúmenes y filtros
# ----------------------------

def summarize_by(DF: pd.DataFrame, cols_group, metric='acc_test'):
    out = DF.groupby(cols_group)[metric].agg(['mean','std','count']).reset_index()
    return out

def filter_by_p(DF: pd.DataFrame, p_values):
    """
    Mantiene WL (p=NaN) y además filas con p en p_values.
    """
    if p_values is None or len(p_values) == 0:
        return DF.copy()
    keep = DF['variant'].eq('WL')
    for p in p_values:
        keep = keep | DF['p'].fillna(-1).eq(p)
    return DF[keep].copy()

def order_variants(DF: pd.DataFrame):
    order = ['WL','dropWL-mean','dropWL+MLP']
    DF['variant'] = pd.Categorical(DF['variant'], categories=order, ordered=True)
    return DF.sort_values(['variant','n'])

# ----------------------------
# Figuras
# ----------------------------

def compute_auto_ylim(df_list, metric='mean', pad=0.02):
    values = []
    for df in df_list:
        if df is None or df.empty:
            continue
        lo = (df['mean'] - df['std'].fillna(0.0)).min()
        hi = (df['mean'] + df['std'].fillna(0.0)).max()
        values.append(lo); values.append(hi)
    if not values:
        return (0.0, 1.0)
    lo = max(0.0, min(values) - pad)
    hi = min(1.0, max(values) + pad)
    if hi - lo < 0.2:  # ampliar rango mínimo
        mid = (hi + lo) / 2
        lo = max(0.0, mid - 0.15)
        hi = min(1.0, mid + 0.15)
    # redondeo a pasos razonables
    lo = max(0.0, np.floor(lo*20)/20.0)
    hi = min(1.0, np.ceil(hi*20)/20.0)
    return (lo, hi)

def plot_mean_std_lines(ax, df_sum, title, ymin=None, ymax=None):
    if df_sum is None or df_sum.empty:
        ax.set_axis_off()
        ax.set_title(title + " (sin datos)")
        return
    for vname, sub in df_sum.groupby('variant'):
        xs = sub['n'].values
        ys = sub['mean'].values
        es = sub['std'].values
        ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=vname)
    ax.set_xlabel('n (número de nodos)')
    ax.set_ylabel('Exactitud')
    if ymin is not None or ymax is not None:
        lo = 0.0 if ymin is None else ymin
        hi = 1.0 if ymax is None else ymax
        ax.set_ylim(lo, hi)
        step = 0.05 if (hi - lo) <= 0.5 else 0.1
        yticks = np.arange(np.ceil(lo/step)*step, hi+1e-9, step)
        ax.set_yticks(np.round(yticks, 2))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(title='Variante', loc='best')
    ax.set_title(title)

def grouped_bars(ax, df_sum, title, ymin=None, ymax=None):
    if df_sum is None or df_sum.empty:
        ax.set_axis_off()
        ax.set_title(title + " (sin datos)")
        return
    variants = list(df_sum['variant'].cat.categories)
    ns = sorted(df_sum['n'].unique())
    width = 0.8 / max(1, len(variants))
    x = np.arange(len(ns))
    for i, v in enumerate(variants):
        sub = df_sum[df_sum['variant'] == v]
        means = [float(sub[sub['n']==n]['mean']) if n in set(sub['n']) else np.nan for n in ns]
        stds  = [float(sub[sub['n']==n]['std'])  if n in set(sub['n']) else np.nan for n in ns]
        ax.bar(x + i*width - 0.5*width*(len(variants)-1), means, width, yerr=stds, capsize=3, label=v)
    ax.set_xticks(x); ax.set_xticklabels(ns)
    ax.set_xlabel('n (número de nodos)')
    ax.set_ylabel('Exactitud (test)')
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin if ymin is not None else 0.0, ymax if ymax is not None else 1.0)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(title='Variante', loc='best')
    ax.set_title(title)

def scatter_train_vs_test(ax, DFp, title):
    if DFp is None or DFp.empty:
        ax.set_axis_off()
        ax.set_title(title + " (sin datos)")
        return
    colors = {'WL':'#1f77b4','dropWL-mean':'#2ca02c','dropWL+MLP':'#d62728'}
    for vname, sub in DFp.groupby('variant'):
        ax.scatter(sub['acc_train'], sub['acc_test'], s=25, alpha=0.6, label=vname, c=colors.get(vname, None))
    ax.plot([0,1],[0,1],'--',color='gray', lw=1)
    ax.set_xlim(0.4, 1.01)
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel('Accuracy train')
    ax.set_ylabel('Accuracy test')
    ax.grid(True, alpha=0.25)
    ax.legend(title='Variante', loc='best')
    ax.set_title(title)

def heatmap(ax, Z, x_ticks, y_ticks, x_label, y_label, title):
    if Z.size == 0:
        ax.set_axis_off()
        ax.set_title(title + " (sin datos)")
        return
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(np.arange(len(x_ticks))); ax.set_xticklabels(x_ticks)
    ax.set_yticks(np.arange(len(y_ticks))); ax.set_yticklabels(y_ticks)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy test')

def build_grid(df, row_key, col_key, val_key):
    # Devuelve matriz con filas = row_key ordenadas, cols = col_key ordenadas
    rows = sorted(df[row_key].dropna().unique())
    cols = sorted(df[col_key].dropna().unique())
    if not rows or not cols:
        return np.array([[]]), [], []
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            sub = df[(df[row_key]==r) & (df[col_key]==c)]
            if not sub.empty:
                M[i,j] = sub['acc_test'].mean()
    return M, rows, cols

def deltas_vs_WL(DFp):
    out = []
    # Comparamos por (n, p, seed): delta = acc(variant) - acc(WL)
    keys = ['n', 'p', 'seed']
    for gkey, sub in DFp.groupby(keys):
        if 'WL' not in set(sub['variant']):
            continue
        wl_val = float(sub[sub['variant']=='WL']['acc_test'].mean())
        for v in ['dropWL-mean','dropWL+MLP']:
            if v in set(sub['variant']):
                val = float(sub[sub['variant']==v]['acc_test'].mean())
                out.append({'n':gkey[0], 'p':gkey[1], 'seed':gkey[2], 'variant':v, 'delta':val - wl_val})
    if not out:
        return pd.DataFrame(columns=['n','p','seed','variant','delta'])
    return pd.DataFrame(out)

def boxplot_by_n_variant(ax, DFp, title):
    if DFp is None or DFp.empty:
        ax.set_axis_off()
        ax.set_title(title + " (sin datos)")
        return
    ns = sorted(DFp['n'].unique())
    variants = list(DFp['variant'].cat.categories)
    # Construimos cajas grouped por variante para cada n → mostramos una por variante y n
    # Para simplificar, apilamos en orden v1(n1..nk), v2(n1..nk), ...
    data = []
    labels = []
    for v in variants:
        for n in ns:
            vals = DFp[(DFp['variant']==v) & (DFp['n']==n)]['acc_test'].values
            if vals.size == 0:
                data.append([])
            else:
                data.append(vals)
            labels.append(f"{v}\n n={n}")
    bp = ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel('Accuracy test')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.2)
    # Rotar etiquetas si hay muchas
    if len(labels) > 12:
        for tick in ax.get_xticklabels():
            tick.set_rotation(75)
            tick.set_ha('right')

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Raíz con c4_n*/**/results.csv')
    ap.add_argument('--out', type=str, required=True, help='PDF multipágina de salida')
    ap.add_argument('--ps', type=float, nargs='+', default=[0.1, 0.2], help='Valores de p a incluir (además de WL)')
    ap.add_argument('--ymin', type=float, default=None, help='Y-min común para curvas/barras')
    ap.add_argument('--ymax', type=float, default=None, help='Y-max común para curvas/barras')
    ap.add_argument('--auto_ylim', action='store_true', help='Calcular automáticamente Y-limits si no se pasan')
    ap.add_argument('--y_pad', type=float, default=0.02, help='Margen para auto_ylim')
    args = ap.parse_args()

    root = Path(args.root)
    out_pdf = Path(args.out)
    out_dir = out_pdf.with_suffix('').with_name(out_pdf.stem + '_assets')
    out_dir.mkdir(parents=True, exist_ok=True)

    DF = load_all_results(root)
    DF = reconstruct_p(DF)
    DF = ensure_columns(DF)
    DF = order_variants(DF)

    # Guardamos un CSV “master” para trazar
    DF.to_csv(out_dir / 'master_all_rows.csv', index=False)

    # Preparamos PDF multipágina
    with PdfPages(out_pdf) as pdf:
        # Por cada p (más WL), hacemos un bloque de figuras
        for p in args.ps:
            DFp = filter_by_p(DF, [p])
            DFp = order_variants(DFp)

            # Resúmenes
            sum_train = summarize_by(DFp, ['variant','n'], metric='acc_train')
            sum_test  = summarize_by(DFp, ['variant','n'], metric='acc_test')

            # Y-limits
            ymin = args.ymin
            ymax = args.ymax
            if args.auto_ylim and (ymin is None and ymax is None):
                ymin, ymax = compute_auto_ylim([sum_train, sum_test], pad=args.y_pad)

            # 1) Curvas mean±std (train/test)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            plot_mean_std_lines(ax[0], sum_train, f'Curvas mean±std (TRAIN) — p={p}', ymin, ymax)
            plot_mean_std_lines(ax[1], sum_test,  f'Curvas mean±std (TEST)  — p={p}', ymin, ymax)
            fig.suptitle(f'Comparativa WL / dropWL-mean / dropWL+MLP — p={p}', y=1.03, fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
            sum_train.to_csv(out_dir / f'summary_train_p{str(p).replace(".","")}.csv', index=False)
            sum_test.to_csv(out_dir  / f'summary_test_p{str(p).replace(".","")}.csv',  index=False)

            # 2) Barras agrupadas (test)
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
            grouped_bars(ax, sum_test, f'Barras agrupadas (TEST) — p={p}', ymin, ymax)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # 3) Scatter train vs test
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
            scatter_train_vs_test(ax, DFp, f'Scatter train vs test — p={p}')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # 4) Heatmap dropWL-mean: acc_test vs (R, t)
            df_mean = DFp[DFp['variant']=='dropWL-mean'].copy()
            have_R = 'dw_R' in df_mean.columns
            have_t = 'dw_t' in df_mean.columns
            if not df_mean.empty and have_R and have_t:
                grid, rows, cols = build_grid(df_mean, row_key='dw_R', col_key='dw_t', val_key='acc_test')
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                heatmap(ax, grid, x_ticks=cols, y_ticks=rows,
                        x_label='t (iteraciones WL)', y_label='R (réplicas)',
                        title=f'dropWL-mean: acc_test vs (R, t) — p={p}')
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

            # 5) Heatmap dropWL+MLP: acc_test vs (layers, d)
            df_mlp = DFp[DFp['variant']=='dropWL+MLP'].copy()
            have_L = 'mlp_layers' in df_mlp.columns
            have_d = 'mlp_d' in df_mlp.columns
            if not df_mlp.empty and have_L and have_d:
                grid, rows, cols = build_grid(df_mlp, row_key='mlp_layers', col_key='mlp_d', val_key='acc_test')
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                heatmap(ax, grid, x_ticks=cols, y_ticks=rows,
                        x_label='d (dimensión latente)', y_label='layers (capas MLP)',
                        title=f'dropWL+MLP: acc_test vs (layers, d) — p={p}')
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

            # 6) Δ vs WL por n
            dlt = deltas_vs_WL(DFp)
            if not dlt.empty:
                fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
                for v, sub in dlt.groupby('variant'):
                    s2 = sub.groupby('n')['delta'].agg(['mean','std']).reset_index()
                    ax.errorbar(s2['n'], s2['mean'], yerr=s2['std'], marker='o', capsize=3, label=v)
                ax.axhline(0.0, color='gray', ls='--', lw=1)
                ax.set_xlabel('n (número de nodos)')
                ax.set_ylabel('Δ Accuracy vs WL')
                ax.grid(True, alpha=0.25)
                ax.legend(title='Variante', loc='best')
                ax.set_title(f'Mejora sobre WL (TEST) — p={p}')
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)
                dlt.to_csv(out_dir / f'deltas_vs_WL_p{str(p).replace(".","")}.csv', index=False)

            # 7) Boxplots acc_test por variante y n
            fig, ax = plt.subplots(1, 1, figsize=(max(7, len(DFp['n'].unique())*1.2), 4.8))
            boxplot_by_n_variant(ax, DFp, title=f'Boxplots acc_test por variante y n — p={p}')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # (Página final) Metadatos del conjunto
        fig, ax = plt.subplots(1, 1, figsize=(7, 0.01))
        plt.axis('off')
        pdf.attach_note("Generado por analyze_extended.py — contiene todas las vistas descritas en el doc.")
        plt.close(fig)

    print(f"[OK] PDF multipágina: {out_pdf}")
    print(f"[OK] CSV maestro: {out_dir/'master_all_rows.csv'}")
    print(f"[OK] Assets (PNGs/CSVs): {out_dir}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
