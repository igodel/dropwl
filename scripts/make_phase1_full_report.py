#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Informe Fase 1 (completo):
- Tabla global TRAIN/TEST (media±DE) por dataset x modelo (5 seeds)
- Barras con error (TEST y TRAIN) por dataset
- Curvas (TRAIN y TEST) por dataset: eje x = seeds (para no-C4)
- Curvas C4 (TRAIN y TEST): eje x = n (8,16,24,32,40,44)
- Violin/box por dataset (TEST) para dispersión por seed
- Scatter train vs test por dataset
- Análisis de tiempos (t_repr, t_fit, t_eval, t_total) por dataset/modelo
- Resumen “ganador por dataset” (TEST) + margen
Lee exclusivamente desde results/fase1_simple_all/*/results.csv
"""
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MODELS = ["WL", "1drop-LOG", "1drop-MLP"]
PALETTE = {"WL":"C0", "1drop-LOG":"C1", "1drop-MLP":"C2"}

DATASETS_ORDER = [
    ("C4_n8","C4 (n=8)"), ("C4_n16","C4 (n=16)"), ("C4_n24","C4 (n=24)"),
    ("C4_n32","C4 (n=32)"), ("C4_n40","C4 (n=40)"), ("C4_n44","C4 (n=44)"),
    ("LIMITS1","LIMITS-1"), ("LIMITS2","LIMITS-2"),
    ("SKIP32","Skip-Circles (n=32)"),
    ("LCC32","LCC (node, n=32)"),
    ("TRI32","Triangles (node, n=32)"),
]
NICE = dict(DATASETS_ORDER)

TIME_COLS = ["t_repr_s","t_fit_s","t_eval_s","t_total_s"]



def load_dataset_dir(ds_dir: Path) -> Optional[pd.DataFrame]:
    """Carga un dataset consolidando results.csv maestro o, si está ‘raro’,
    concatena results.csv por-seed bajo subcarpetas de cada modelo."""
    master = ds_dir/"results.csv"
    if not master.exists():
        return None
    try:
        dfm = pd.read_csv(master)
    except Exception:
        dfm = pd.DataFrame()

    need = {"variant","seed","acc_train","acc_test"}
    if not need.issubset(dfm.columns):
        # Recuperación desde per-seed:
        rows = []
        for m in MODELS:
            mdir = ds_dir/m
            if not mdir.exists(): 
                continue
            for sdir in sorted(mdir.glob("seed_*")):
                f = sdir/"results.csv"
                if f.exists():
                    try:
                        rows.append(pd.read_csv(f))
                    except Exception:
                        pass
        if rows:
            dfm = pd.concat(rows, ignore_index=True)
        else:
            raise RuntimeError(f"results.csv sin columnas requeridas en {ds_dir}")

    # Filtrar a modelos válidos y estandarizar tipos
    dfm = dfm[dfm["variant"].isin(MODELS)].copy()
    for c in ["acc_train","acc_test"]+TIME_COLS:
        if c in dfm.columns:
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
    # seed a int si es posible
    if "seed" in dfm.columns:
        dfm["seed"] = pd.to_numeric(dfm["seed"], errors="coerce").astype("Int64")

    # Quitar filas con NaN en métricas principales
    dfm = dfm.dropna(subset=["acc_train","acc_test"], how="any")
    return dfm if len(dfm) else None

def agg_table(df: pd.DataFrame, col="acc_test") -> pd.DataFrame:
    g = df.groupby("variant")[col].agg(["mean","std","count"])
    return g.reindex(MODELS)

def draw_table_global(pdf: PdfPages, BIG: pd.DataFrame, split: str):
    fig, ax = plt.subplots(figsize=(12, 0.55*len(DATASETS_ORDER)+1.5))
    ax.axis("off")
    ax.set_title(f"Tabla global {split} · media ± DE (5 seeds)", pad=12)
    # construir tabla bonita dataset x modelo
    rows = {}
    sub = BIG[BIG["split"]==split]
    for ds,_ in DATASETS_ORDER:
        row = {}
        sdf = sub[sub["dataset"]==ds]
        for m in MODELS:
            r = sdf[sdf["variant"]==m]
            if len(r):
                mu, sd = r["mean"].values[0], r["std"].values[0]
                row[m] = f"{mu:.3f} ± {sd:.3f}"
            else:
                row[m] = "–"
        rows[NICE[ds]] = row
    tbl = pd.DataFrame(rows).T[MODELS]
    the_table = ax.table(cellText=tbl.values, rowLabels=tbl.index,
                         colLabels=tbl.columns, loc="center")
    the_table.auto_set_font_size(False); the_table.set_fontsize(9); the_table.scale(1,1.2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def bars_with_error(pdf: PdfPages, df: pd.DataFrame, ds_key: str, which="acc_test"):
    ag = agg_table(df, which)
    fig, ax = plt.subplots(figsize=(6.8,3.6))
    x = np.arange(len(MODELS))
    means = ag["mean"].values
    stds  = ag["std"].fillna(0.0).values
    ax.bar(x, means, yerr=stds, capsize=4, color=[PALETTE[m] for m in MODELS])
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylim(0.45, 1.02)
    lbl = "TEST" if which=="acc_test" else "TRAIN"
    ax.set_ylabel(f"Accuracy ({lbl})"); ax.set_title(f"{NICE[ds_key]} · {lbl} (media±DE)")
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def lines_by_seed(pdf: PdfPages, df: pd.DataFrame, ds_key: str):
    """Curvas TRAIN y TEST por seed, 3 modelos (válido para todos los datasets)."""
    if "seed" not in df.columns or df["seed"].isna().any():
        return
    seeds = sorted(df["seed"].dropna().unique())
    fig, axes = plt.subplots(2,1, figsize=(7.2,6.0), sharex=True)
    for r, col, ttl in [(0,"acc_test","TEST"), (1,"acc_train","TRAIN")]:
        ax = axes[r]
        for m in MODELS:
            d = df[df["variant"]==m].copy()
            y = [d[d["seed"]==s][col].mean() for s in seeds]  # media si hay múltiples splits
            ax.plot(seeds, y, marker="o", label=m, color=PALETTE[m])
        ax.set_ylabel(f"Accuracy ({ttl})")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
        ax.set_title(f"{NICE[ds_key]} · curvas por seed ({ttl})")
    axes[-1].set_xlabel("Seed")
    axes[0].legend(loc="lower right")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def c4_curves_n(pdf: PdfPages, bundle: dict):
    """Curvas C4: n -> accuracy, para TEST y TRAIN."""
    keys = [k for k,_ in DATASETS_ORDER if k.startswith("C4_n") and k in bundle]
    if not keys: return
    rows_test, rows_train = [], []
    for k in keys:
        n = int(k.split("_n")[1])
        ag_te = agg_table(bundle[k],"acc_test").reset_index(); ag_te["n"]=n
        ag_tr = agg_table(bundle[k],"acc_train").reset_index(); ag_tr["n"]=n
        rows_test.append(ag_te); rows_train.append(ag_tr)
    TE = pd.concat(rows_test, ignore_index=True)
    TR = pd.concat(rows_train, ignore_index=True)
    for which, dfc, title in [("TEST", TE, "Familia C4 · TEST"),
                              ("TRAIN",TR, "Familia C4 · TRAIN")]:
        fig, ax = plt.subplots(figsize=(7.2,3.8))
        for m in MODELS:
            sub = dfc[dfc["variant"]==m].sort_values("n")
            ax.plot(sub["n"], sub["mean"], marker="o", label=m, color=PALETTE[m])
            lo = (sub["mean"]-sub["std"].fillna(0)).values
            hi = (sub["mean"]+sub["std"].fillna(0)).values
            ax.fill_between(sub["n"], lo, hi, color=PALETTE[m], alpha=0.15)
        ax.set_xlabel("n (nodos)")
        ax.set_ylabel(f"Accuracy ({which})")
        ax.set_ylim(0.5, 1.02)
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def violin_box(pdf: PdfPages, df: pd.DataFrame, ds_key: str):
    """Distribución (TEST) por modelo (violin + box)"""
    if "seed" not in df.columns or df["seed"].isna().any():
        return
    fig, ax = plt.subplots(figsize=(6.8,3.6))
    data = [df[df["variant"]==m]["acc_test"].values for m in MODELS]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts['bodies']: pc.set_alpha(0.3)
    ax.boxplot(data, widths=0.12, showfliers=False)
    ax.set_xticks(np.arange(1,len(MODELS)+1)); ax.set_xticklabels(MODELS)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Accuracy (TEST)")
    ax.set_title(f"{NICE[ds_key]} · dispersión por seed (TEST)")
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def scatter_train_test(pdf: PdfPages, df: pd.DataFrame, ds_key: str):
    fig, ax = plt.subplots(figsize=(6.4,3.8))
    for m in MODELS:
        d = df[df["variant"]==m]
        ax.scatter(d["acc_train"], d["acc_test"], s=28, alpha=0.9, label=m, color=PALETTE[m])
    ax.plot([0.45,1.0],[0.45,1.0],"k--", lw=1, alpha=0.5)
    ax.set_xlim(0.45,1.02); ax.set_ylim(0.45,1.02)
    ax.set_xlabel("Accuracy (TRAIN)"); ax.set_ylabel("Accuracy (TEST)")
    ax.set_title(f"{NICE[ds_key]} · scatter train vs test (puntos=seeds)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def times_analysis(pdf: PdfPages, df: pd.DataFrame, ds_key: str):
    # Medias de tiempos por modelo
    have = [c for c in TIME_COLS if c in df.columns]
    if not have: return
    ag = df.groupby("variant")[have].mean().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7.2,3.6))
    x = np.arange(len(MODELS))
    width = 0.2
    off = -width
    for c in ["t_repr_s","t_fit_s","t_eval_s","t_total_s"]:
        if c not in ag: continue
        ax.bar(x+off, ag[c].values, width, label=c)
        off += width
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel("Tiempo medio (s)")
    ax.set_title(f"{NICE[ds_key]} · tiempos por etapa (media en seeds)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def winners_summary(pdf: PdfPages, bundle: dict):
    rows=[]
    for ds,_ in DATASETS_ORDER:
        if ds not in bundle: continue
        ag = agg_table(bundle[ds], "acc_test")
        best = ag["mean"].idxmax()
        margin = float(ag.loc[best,"mean"] - ag.drop(index=best)["mean"].max())
        rows.append({"dataset":NICE[ds],
                     "winner":best,
                     "mean":float(ag.loc[best,"mean"]),
                     "std":float(ag.loc[best,"std"]),
                     "margin_vs_2nd":margin})
    R = pd.DataFrame(rows).sort_values("dataset")
    fig, ax = plt.subplots(figsize=(10, 0.55*len(R)+1.2))
    ax.axis("off"); ax.set_title("Resumen final (TEST): mejor modelo por dataset", pad=12)
    table = ax.table(
        cellText=np.column_stack([
            R["winner"].values,
            [f"{m:.3f} ± {s:.3f}" for m,s in zip(R["mean"],R["std"])],
            [f"{m:.3f}" for m in R["margin_vs_2nd"]]
        ]),
        rowLabels=R["dataset"].values,
        colLabels=["Modelo ganador","Accuracy (media±DE)","Margen vs 2º"],
        loc="center"
    )
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Carpeta con subdirs por dataset y su results.csv")
    ap.add_argument("--out",  required=True, help="Nombre del PDF de salida")
    args = ap.parse_args()

    root = Path(args.root)
    out_pdf = Path(args.out)

    # Cargar todo
    bundle = {}
    for ds,_ in DATASETS_ORDER:
        ds_dir = root/ds
        if not ds_dir.exists(): 
            continue
        df = load_dataset_dir(ds_dir)
        if df is not None and len(df):
            df = df.copy(); df["dataset"] = ds
            bundle[ds] = df

    if not bundle:
        raise SystemExit(f"No se pudo cargar nada desde {root}")

    # Construir BIG para tablas globales
    big_rows = []
    for ds, _ in DATASETS_ORDER:
        if ds not in bundle: continue
        for split, col in [("TEST","acc_test"), ("TRAIN","acc_train")]:
            ag = agg_table(bundle[ds], col).reset_index()
            ag["dataset"] = ds; ag["split"] = split
            big_rows.append(ag)
    BIG = pd.concat(big_rows, ignore_index=True)

    # PDF
    with PdfPages(out_pdf) as pdf:
        # Portada simple
        fig, ax = plt.subplots(figsize=(8.5,3))
        ax.axis("off")
        ax.set_title("Informe Fase 1 (Full) — WL vs 1-dropWL-LOG vs 1-dropWL-MLP", pad=16, fontsize=14)
        ax.text(0.01, 0.65, f"Fuente: {root}", fontsize=10)
        ax.text(0.01, 0.45, "Incluye TRAIN & TEST, barras con error, curvas por seed, curvas n→accuracy para C4,\n"
                            "violin/box, scatter train vs test y tiempos por etapa.", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Tablas globales TEST y TRAIN
        draw_table_global(pdf, BIG, "TEST")
        draw_table_global(pdf, BIG, "TRAIN")

        # Curvas C4 n→accuracy (TEST y TRAIN)
        c4_curves_n(pdf, bundle)

        # Por dataset: barras TEST y TRAIN + curvas por seed (TRAIN y TEST)
        for ds,_ in DATASETS_ORDER:
            if ds not in bundle: continue
            df = bundle[ds]
            bars_with_error(pdf, df, ds, "acc_test")
            bars_with_error(pdf, df, ds, "acc_train")
            lines_by_seed(pdf, df, ds)
            violin_box(pdf, df, ds)
            scatter_train_test(pdf, df, ds)
            times_analysis(pdf, df, ds)

        # Resumen final “ganador”
        winners_summary(pdf, bundle)

    # CSVs auxiliares globales (por si quieres tabular en LaTeX)
    out_dir = out_pdf.parent
    BIG.to_csv(out_dir/"phase1_full_global_table.csv", index=False)
    # Tiempos promediados
    time_rows = []
    for ds,_ in DATASETS_ORDER:
        if ds not in bundle: continue
        df = bundle[ds]
        ag = df.groupby("variant")[TIME_COLS].mean() if set(TIME_COLS).issubset(df.columns) else pd.DataFrame()
        if len(ag):
            ag = ag.reindex(MODELS).reset_index()
            ag.insert(0, "dataset", ds)
            time_rows.append(ag)
    if time_rows:
        TIMES = pd.concat(time_rows, ignore_index=True)
        TIMES.to_csv(out_dir/"phase1_full_times.csv", index=False)

    print(f"[OK] PDF: {out_pdf}")
    print(f"[OK] CSV global: {out_dir/'phase1_full_global_table.csv'}")
    if time_rows:
        print(f"[OK] CSV tiempos: {out_dir/'phase1_full_times.csv'}")

if __name__ == "__main__":
    main()
