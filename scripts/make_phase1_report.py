# scripts/make_phase1_report.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

DATASETS_ORDER = [
    "C4_n8","C4_n16","C4_n24","C4_n32","C4_n40","C4_n44",
    "LIMITS1","LIMITS2","SKIP32","LCC32","TRI32"
]
VARIANTS_ORDER = ["WL","1drop-LOG","1drop-MLP"]
VARIANT_LABELS = {"WL":"WL","1drop-LOG":"1-dropWL-LOG","1drop-MLP":"1-dropWL-MLP"}

def load_master(root: Path) -> pd.DataFrame:
    """Concatena todos los results.csv por dataset en root/<DATASET>/results.csv."""
    rows=[]
    for ds in DATASETS_ORDER:
        f = root/ds/"results.csv"
        if not f.exists():
            print(f"[WARN] falta {f}")
            continue
        df = pd.read_csv(f)
        df["dataset"] = ds
        rows.append(df)
    if not rows:
        raise SystemExit("[FATAL] No se encontraron results.csv")
    DF = pd.concat(rows, ignore_index=True)
    # columnas obligatorias
    need = {"dataset","variant","acc_train","acc_test","seed"}
    miss = need - set(DF.columns)
    if miss:
        raise SystemExit(f"[FATAL] Faltan columnas en master: {miss}")
    return DF

def fmt_mean_std(g):
    m = g.mean()
    s = g.std(ddof=0)  # población; da igual para 5 seeds
    return f"{m:.4f} ± {s:.4f}"

def summary_table(DF: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen: filas=datasets, cols=modelos; celda = mean±std (test)."""
    out = pd.DataFrame(index=DATASETS_ORDER, columns=VARIANTS_ORDER)
    for ds in DATASETS_ORDER:
        sub = DF[DF["dataset"]==ds]
        for v in VARIANTS_ORDER:
            sv = sub[sub["variant"]==v]["acc_test"]
            out.loc[ds, v] = fmt_mean_std(sv) if len(sv)>0 else "–"
    # Renombrar columnas a etiquetas “bonitas”
    out = out.rename(columns=VARIANT_LABELS)
    return out

def draw_table_page(pdf: PdfPages, table_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10.5, 7.5))  # US Letter-like
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=12)
    tbl = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def bars_by_dataset(pdf: PdfPages, DF: pd.DataFrame):
    """Una página por dataset: barras (test) WL vs 1-dropLOG vs 1-dropMLP con mean±std."""
    for ds in DATASETS_ORDER:
        sub = DF[DF["dataset"]==ds]
        means, stds, labels = [], [], []
        for v in VARIANTS_ORDER:
            x = sub[sub["variant"]==v]["acc_test"]
            if len(x)==0: continue
            means.append(x.mean()); stds.append(x.std(ddof=0)); labels.append(VARIANT_LABELS[v])
        if not means: continue
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        idx = np.arange(len(means))
        ax.bar(idx, means, yerr=stds, capsize=4)
        ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Accuracy (test)")
        ax.set_title(f"{ds}: test mean±std")
        ax.grid(axis='y', alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def curves_c4(pdf: PdfPages, DF: pd.DataFrame):
    """Curvas mean±std vs n para C4 (train y test)."""
    c4_list = ["C4_n8","C4_n16","C4_n24","C4_n32","C4_n40","C4_n44"]
    ns      = [8,16,24,32,40,44]
    for split_col, title in [("acc_train","Curvas mean±std (TRAIN) C4"),
                             ("acc_test","Curvas mean±std (TEST)  C4")]:
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        for v in VARIANTS_ORDER:
            means, stds = [], []
            for ds in c4_list:
                x = DF[(DF["dataset"]==ds) & (DF["variant"]==v)][split_col]
                means.append(x.mean() if len(x)>0 else np.nan)
                stds.append(x.std(ddof=0) if len(x)>0 else np.nan)
            means = np.array(means); stds = np.array(stds)
            ax.plot(ns, means, marker='o', label=VARIANT_LABELS[v])
            ax.fill_between(ns, means-stds, means+stds, alpha=0.15)
        ax.set_xlabel("n (número de nodos)")
        ax.set_ylabel("Exactitud")
        ax.set_ylim(0.55, 1.01)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def boxplots_c4(pdf: PdfPages, DF: pd.DataFrame):
    """Boxplots (test) por (variante×n) para C4."""
    c4_list = ["C4_n8","C4_n16","C4_n24","C4_n32","C4_n40","C4_n44"]
    labels, data = [], []
    for ds in c4_list:
        for v in VARIANTS_ORDER:
            x = DF[(DF["dataset"]==ds) & (DF["variant"]==v)]["acc_test"]
            if len(x):
                labels.append(f"{VARIANT_LABELS[v]}\n{ds.split('_')[1]}")
                data.append(x.values)
    if not data: return
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylim(0.55, 1.01)
    ax.set_ylabel("Accuracy (test)")
    ax.set_title("C4: boxplots (test) por variante y n")
    plt.xticks(rotation=0)
    ax.grid(axis='y', alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def text_page(pdf: PdfPages, header: str, body: str):
    fig, ax = plt.subplots(figsize=(8.3, 11.7))  # A4 portrait-like
    ax.axis('off')
    ax.text(0.5, 0.95, header, ha='center', va='top', fontsize=18, weight='bold')
    wrapped = "\n".join(textwrap.wrap(body, width=100))
    ax.text(0.06, 0.90, wrapped, ha='left', va='top', fontsize=11)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Carpeta con results.csv por dataset (Fase 1)")
    ap.add_argument("--out",  type=str, default="report_phase1_simple.pdf")
    args = ap.parse_args()

    root = Path(args.root)
    DF = load_master(root)

    # Tabla maestra (test)
    table = summary_table(DF)

    with PdfPages(args.out) as pdf:
        # Portada breve
        text_page(pdf, "Fase 1 – Modelos simples (WL / 1-dropWL-LOG / 1-dropWL-MLP)",
                  "Protocolo: 11 datasets sintéticos, 5 seeds, split 70/30, estandarización activada, "
                  "WL (t=3, kmax=40), 1-drop (R=50). MLP: layers=2, hidden=128, d=64, act=relu. "
                  "Métrica: accuracy (train/test). Se reportan medias y desviaciones estándar sobre 5 seeds.")

        # Tabla maestra (una página)
        draw_table_page(pdf, table, "Tabla global (test): media ± DE por dataset × modelo")

        # Curvas C4 (train y test)
        curves_c4(pdf, DF)

        # Boxplots C4 (test)
        boxplots_c4(pdf, DF)

        # Barras por dataset (test)
        bars_by_dataset(pdf, DF)

    print(f"[OK] PDF: {args.out}")

if __name__ == "__main__":
    main()
