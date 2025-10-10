PYTHONPATH=. python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

root = Path("results/fase1_simple_all")
out_pdf = Path("report_phase1_full.pdf")

# Orden lógico de datasets y etiquetas bonitas
ds_order = [
    ("C4_n8","C4 (n=8)"), ("C4_n16","C4 (n=16)"), ("C4_n24","C4 (n=24)"),
    ("C4_n32","C4 (n=32)"), ("C4_n40","C4 (n=40)"), ("C4_n44","C4 (n=44)"),
    ("LIMITS1","LIMITS-1"), ("LIMITS2","LIMITS-2"),
    ("SKIP32","Skip-Circles (n=32)"),
    ("LCC32","LCC (node, n=32)"),
    ("TRI32","Triangles (node, n=32)")
]
nice = dict(ds_order)

# Modelos en el orden pedido
model_order = ["WL", "1drop-LOG", "1drop-MLP"]
palette = {"WL":"C0","1drop-LOG":"C1","1drop-MLP":"C2"}

# ---- Cargar/conciliar todos los datasets ----
def load_dataset_dir(ds_dir: Path):
    # Lee maestro results.csv
    master = ds_dir/"results.csv"
    if not master.exists():
        return None
    dfm = pd.read_csv(master)

    # Normalizamos columnas esperadas
    # Deben existir: ['variant','seed','acc_train','acc_test'] (tu exp_simple_compare las escribe)
    needed = {"variant","seed","acc_train","acc_test"}
    missing = needed - set(dfm.columns)
    if missing:
        # intentar rescatar por-seed si el maestro está raro
        rows = []
        for v in model_order:
            for seed_dir in (ds_dir/v).glob("seed_*"):
                f = seed_dir/"results.csv"
                if f.exists():
                    rows.append(pd.read_csv(f))
        if rows:
            dfm = pd.concat(rows, ignore_index=True)
        else:
            raise RuntimeError(f"columns missing in {master}: {missing}")

    # Filtrar a solo los modelos de interés
    dfm = dfm[dfm["variant"].isin(model_order)].copy()

    # Asegurar tipos
    for c in ["acc_train","acc_test"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
    dfm = dfm.dropna(subset=["acc_train","acc_test"])

    return dfm

all_data = {}
for ds, _ in ds_order:
    ds_dir = root/ds
    if ds_dir.is_dir():
        try:
            df = load_dataset_dir(ds_dir)
            if df is not None and len(df):
                df["dataset"] = ds
                all_data[ds] = df
        except Exception as e:
            print(f"[WARN] no se pudo leer {ds_dir}: {e}")

assert all_data, "No se cargó ningún dataset."

# ---- Helper: tabla resumen mean±std ----
def agg_table(df, col="acc_test"):
    g = df.groupby("variant")[col].agg(["mean","std","count"])
    g = g.reindex(model_order)
    return g

# ---- PDF ----
with PdfPages(out_pdf) as pdf:

    # 0) Portada / resumen global (TEST y TRAIN)
    big_rows = []
    for ds,_ in ds_order:
        if ds in all_data:
            df = all_data[ds]
            for split, col in [("TEST","acc_test"),("TRAIN","acc_train")]:
                ag = agg_table(df, col)
                ag["dataset"] = ds
                ag["split"] = split
                big_rows.append(ag.reset_index())
    BIG = pd.concat(big_rows, ignore_index=True)

    # Tabla global (una por split)
    for split in ["TEST","TRAIN"]:
        fig, ax = plt.subplots(figsize=(12, 0.55*len(ds_order)+1.5))
        sub = BIG[BIG["split"]==split].copy()
        # Construye tabla pivot: filas=datasets, cols=modelos, celdas=mean±std (texto)
        pivot = {}
        for ds,_ in ds_order:
            row = {}
            dfds = sub[sub["dataset"]==ds]
            for m in model_order:
                r = dfds[dfds["variant"]==m]
                if len(r):
                    mu = r["mean"].values[0]
                    sd = r["std"].values[0]
                    row[m] = f"{mu:.3f} ± {sd:.3f}"
                else:
                    row[m] = "–"
            pivot[nice[ds]] = row
        tbl = pd.DataFrame(pivot).T[model_order]
        ax.axis("off")
        ax.set_title(f"Tabla global {split} · media ± DE (5 seeds)", pad=12)
        the_table = ax.table(
            cellText=tbl.values,
            rowLabels=tbl.index,
            colLabels=tbl.columns,
            loc="center"
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # 1) Barras con error (TEST) por dataset
    for ds,_ in ds_order:
        if ds not in all_data: continue
        df = all_data[ds]
        ag = agg_table(df, "acc_test")
        fig, ax = plt.subplots(figsize=(6.5,3.6))
        x = np.arange(len(model_order))
        means = ag["mean"].values
        stds  = ag["std"].values
        ax.bar(x, means, yerr=stds, capsize=4, color=[palette[m] for m in model_order])
        ax.set_xticks(x); ax.set_xticklabels(model_order)
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("Accuracy (TEST)")
        ax.set_title(f"{nice[ds]} · TEST (media±DE, 5 seeds)")
        ax.grid(axis="y", alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # 2) Curvas (n vs accuracy) solo para C4_*
    C4_keys = [k for k,_ in ds_order if k.startswith("C4_n")]
    if all(k in all_data for k in C4_keys):
        # Construir frame con mean±std por n y variante
        rows=[]
        for k in C4_keys:
            n = int(k.split("_n")[1])
            ag = agg_table(all_data[k], "acc_test")
            ag["n"] = n
            ag["dataset"] = k
            rows.append(ag.reset_index())
        C4 = pd.concat(rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(6.8,3.6))
        for m in model_order:
            sub = C4[C4["variant"]==m].sort_values("n")
            ax.plot(sub["n"], sub["mean"], marker="o", label=m, color=palette[m])
            ax.fill_between(sub["n"], sub["mean"]-sub["std"], sub["mean"]+sub["std"],
                            alpha=0.15, color=palette[m])
        ax.set_xlabel("n (nodos)")
        ax.set_ylabel("Accuracy (TEST)")
        ax.set_ylim(0.5, 1.02)
        ax.set_title("Familia C4 · mean±std (TEST) sobre 5 seeds")
        ax.legend()
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # 3) Violin/box + swarm (TEST) por dataset (dispersión por seed)
    for ds,_ in ds_order:
        if ds not in all_data: continue
        df = all_data[ds]
        if "seed" not in df.columns: continue
        fig, ax = plt.subplots(figsize=(6.8,3.6))
        data = [df[df["variant"]==m]["acc_test"].values for m in model_order]
        # violin
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.3)
        # box encima
        ax.boxplot(data, widths=0.1, showfliers=False)
        ax.set_xticks(np.arange(1,len(model_order)+1))
        ax.set_xticklabels(model_order)
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("Accuracy (TEST)")
        ax.set_title(f"{nice[ds]} · dispersión por seed (TEST)")
        ax.grid(axis="y", alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # 4) Scatter train vs test por dataset
    for ds,_ in ds_order:
        if ds not in all_data: continue
        df = all_data[ds].copy()
        fig, ax = plt.subplots(figsize=(6.1,3.6))
        for m in model_order:
            d = df[df["variant"]==m]
            ax.scatter(d["acc_train"], d["acc_test"], s=28, label=m, alpha=0.85, color=palette[m])
        ax.plot([0.45,1.0],[0.45,1.0],"k--",lw=1,alpha=0.5)
        ax.set_xlim(0.45,1.02); ax.set_ylim(0.45,1.02)
        ax.set_xlabel("Accuracy (TRAIN)")
        ax.set_ylabel("Accuracy (TEST)")
        ax.set_title(f"{nice[ds]} · scatter train vs test (puntos=seeds)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # 5) Resumen final: mejor modelo por dataset (TEST)
    rows=[]
    for ds,_ in ds_order:
        if ds not in all_data: continue
        ag = agg_table(all_data[ds], "acc_test")
        best = ag["mean"].idxmax()
        margin = ag.loc[best,"mean"] - ag.drop(index=best)["mean"].max()
        rows.append({"dataset":nice[ds],
                     "winner":best,
                     "mean":ag.loc[best,"mean"],
                     "std":ag.loc[best,"std"],
                     "margin_vs_2nd":margin})
    R = pd.DataFrame(rows).sort_values("dataset")
    fig, ax = plt.subplots(figsize=(10, 0.55*len(R)+1.2))
    ax.axis("off")
    ax.set_title("Resumen final (TEST): mejor modelo por dataset", pad=12)
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
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1,1.2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"[OK] PDF: {out_pdf}")
PY
