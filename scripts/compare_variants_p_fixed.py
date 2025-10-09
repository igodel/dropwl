#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparativa de variantes (WL, dropWL-mean, dropWL+MLP) a p fijo (0.1 ó 0.2).
- Lee results/paper_c4_full/c4_n*/**/results.csv
- Reconstruye la columna 'p' si falta (WL=NaN; dropWL-mean <- dw_p; dropWL+MLP <- mlp_p)
- Filtra por p deseado (mantiene WL para referencia)
- Genera curvas mean±std vs n para acc_train y acc_test
- Permite fijar Y-limits (--ymin/--ymax) o calcularlos automáticamente (--auto_ylim)
- Exporta PNGs y un PDF; puedes añadir un sufijo a los nombres (--suffix)

Uso típico:
PYTHONPATH=. python scripts/compare_variants_p_fixed.py \
  --root results/paper_c4_full \
  --p 0.1 \
  --out report_compare_p01_rescaled.pdf \
  --auto_ylim \
  --suffix rescaled
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_all_results(root: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(root.glob('c4_n*/**/results.csv')):
        try:
            df = pd.read_csv(csv)
            n = int(csv.parts[csv.parts.index('paper_c4_full')+1].split('c4_n')[1])
            df['n'] = n
            df['variant_dir'] = csv.parent.name
            rows.append(df)
        except Exception as e:
            print("[WARN] no pude leer:", csv, repr(e))
    if not rows:
        raise RuntimeError(f"No encontré results.csv bajo {root}")
    DF = pd.concat(rows, ignore_index=True)
    return DF

def reconstruct_p_column(DF: pd.DataFrame) -> pd.DataFrame:
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
        # WL queda NaN
    return DF

def filter_for_p(DF: pd.DataFrame, p: float) -> pd.DataFrame:
    keep = DF['variant'].eq('WL') | DF['p'].fillna(-1).eq(p)
    sub = DF[keep].copy()
    sub['p_label'] = sub['p'].astype(str)
    sub.loc[sub['variant']=='WL', 'p_label'] = 'WL'
    return sub

def summarize_by_n_variant(DF: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = DF.groupby(['variant','n'])[metric].agg(['mean','std','count']).reset_index()
    return g.sort_values(['variant','n'])

def _auto_ylim(sum_train: pd.DataFrame, sum_test: pd.DataFrame):
    """
    Calcula límites Y sensatos en base a min(mean-std) y max(mean+std)
    de train y test, con un margen pequeño.
    """
    mins = []
    maxs = []
    for df in (sum_train, sum_test):
        if df.empty:
            continue
        mins.append((df['mean'] - df['std'].fillna(0.0)).min())
        maxs.append((df['mean'] + df['std'].fillna(0.0)).max())
    if not mins or not maxs:
        return (0.0, 1.0)
    lo = min(mins) - 0.02
    hi = max(maxs) + 0.02
    lo = max(0.0, np.floor(lo*20)/20.0)  # redondeo a pasos de 0.05
    hi = min(1.0, np.ceil(hi*20)/20.0)
    # Evitar rangos minúsculos
    if hi - lo < 0.2:
        mid = (hi+lo)/2
        lo = max(0.0, mid - 0.15)
        hi = min(1.0, mid + 0.15)
    return (lo, hi)

def plot_curves(df_sum: pd.DataFrame, metric: str, out_png: Path,
                title: str, ymin: float=None, ymax: float=None):
    """Curvas mean±std vs n (una línea por variante) con control de Y-limits."""
    plt.figure(figsize=(7.0, 4.5))
    for vname, sub in df_sum.groupby('variant'):
        xs = sub['n'].values
        ys = sub['mean'].values
        es = sub['std'].values
        plt.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=vname)
    plt.xlabel('n (número de nodos)')
    plt.ylabel('Exactitud')  # nombre más claro en el eje Y
    if ymin is not None or ymax is not None:
        lo = 0.0 if ymin is None else ymin
        hi = 1.0 if ymax is None else ymax
        plt.ylim(lo, hi)
        # ticks bonitos cada 0.05–0.1 según rango
        step = 0.05 if (hi - lo) <= 0.5 else 0.1
        yticks = np.arange(np.ceil(lo/step)*step, hi+1e-9, step)
        plt.yticks(np.round(yticks, 2))
    else:
        plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(title='Variante', loc='best')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Raíz con c4_n*/**/results.csv')
    ap.add_argument('--p', type=float, required=True, help='p fijo para comparar (0.1 o 0.2)')
    ap.add_argument('--out', type=str, required=True, help='PDF de salida')
    ap.add_argument('--suffix', type=str, default='', help='Sufijo opcional para nombres de PNGs')
    ap.add_argument('--ymin', type=float, default=None, help='Límite inferior Y (opcional)')
    ap.add_argument('--ymax', type=float, default=None, help='Límite superior Y (opcional)')
    ap.add_argument('--auto_ylim', action='store_true', help='Calcular Y-limits automáticamente')
    args = ap.parse_args()

    root = Path(args.root)
    out_pdf = Path(args.out)
    suffix = (('_' + args.suffix) if args.suffix else '')

    DF = load_all_results(root)
    DF = reconstruct_p_column(DF)
    DFp = filter_for_p(DF, args.p)

    order = ['WL', 'dropWL-mean', 'dropWL+MLP']
    DFp['variant'] = pd.Categorical(DFp['variant'], categories=order, ordered=True)

    sum_train = summarize_by_n_variant(DFp, 'acc_train')
    sum_test  = summarize_by_n_variant(DFp, 'acc_test')

    # Determinar Y-limits
    ymin = args.ymin
    ymax = args.ymax
    if args.auto_ylim and (ymin is None and ymax is None):
        ymin, ymax = _auto_ylim(sum_train, sum_test)

    # Nombres de figuras
    ptag = str(args.p).replace('.', '')
    png_train = out_pdf.with_name(out_pdf.stem.replace('.pdf','') + f"_train_p{ptag}{suffix}.png")
    png_test  = out_pdf.with_name(out_pdf.stem.replace('.pdf','') + f"_test_p{ptag}{suffix}.png")

    plot_curves(sum_train, 'acc_train', png_train,
                title=f'Accuracy (train) vs n — p={args.p}',
                ymin=ymin, ymax=ymax)
    plot_curves(sum_test,  'acc_test',  png_test,
                title=f'Accuracy (test)  vs n — p={args.p}',
                ymin=ymin, ymax=ymax)

    with PdfPages(out_pdf) as pdf:
        for img_path, title in [(png_train, 'Train'), (png_test, 'Test')]:
            import matplotlib.image as mpimg
            if Path(img_path).exists():
                img = mpimg.imread(img_path)
                fig = plt.figure(figsize=(7.0, 4.5))
                plt.imshow(img); plt.axis('off'); plt.title(title)
                pdf.savefig(fig); plt.close(fig)

    tbl_dir = out_pdf.with_suffix('').with_name(out_pdf.stem + '_tables')
    tbl_dir.mkdir(exist_ok=True, parents=True)
    sum_train.to_csv(tbl_dir / f'summary_train_p{ptag}.csv', index=False)
    sum_test.to_csv(tbl_dir  / f'summary_test_p{ptag}.csv',  index=False)

    print(f"[OK] PNG (train): {png_train}")
    print(f"[OK] PNG (test) : {png_test}")
    print(f"[OK] PDF        : {out_pdf}")
    print(f"[OK] Tablas CSV : {tbl_dir}")
    if ymin is not None or ymax is not None:
        print(f"[info] Y-limits: ymin={ymin}, ymax={ymax}")

if __name__ == '__main__':
    main()
