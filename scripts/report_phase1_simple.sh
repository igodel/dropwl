PYTHONPATH=. MPLBACKEND=Agg python - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

root = Path("results/fase1_simple_all")
out_pdf = root / "report_phase1_simple.pdf"   # <-- salida forzada AQUÍ
figdir  = root / "figs"
figdir.mkdir(parents=True, exist_ok=True)

DATASETS = ["C4_n8","C4_n16","C4_n24","C4_n32","C4_n40","C4_n44",
            "LIMITS1","LIMITS2","SKIP32","LCC32","TRI32"]

dfs = {}
for d in DATASETS:
    f = root/d/"results.csv"
    if f.exists():
        df = pd.read_csv(f)
        for col in ["variant","seed","acc_test","acc_train","t_total_s","dim"]:
            if col not in df.columns: df[col] = np.nan
        dfs[d] = df

def summary_table(df):
    g = df.groupby("variant")
    return pd.DataFrame({
        "acc_test_mean": g["acc_test"].mean(),
        "acc_test_std":  g["acc_test"].std(),
        "acc_train_mean": g["acc_train"].mean(),
        "acc_train_std":  g["acc_train"].std(),
        "count": g["seed"].nunique()
    }).reset_index()

def fmt_pm(mu, sig):
    if pd.isna(mu): return "–"
    if pd.isna(sig) or sig==0: return f"{mu:.3f}"
    return f"{mu:.3f} ± {sig:.3f}"

with PdfPages(out_pdf) as pdf:
    # Portada
    plt.figure(figsize=(8.5, 11)); plt.axis("off")
    title = "Fase 1 — Informe consolidado (WL, 1-dropWL-LOG, 1-dropWL-MLP)"
    subtitle = "Split 70/30, 5 seeds, kmax=40, t=3, R=50 (1-drop), standardize=True, device=cpu"
    ds_ok = [d for d in DATASETS if d in dfs]
    plt.text(0.05, 0.9, title, fontsize=18, weight="bold")
    plt.text(0.05, 0.86, subtitle, fontsize=10)
    y = 0.82
    for line in textwrap.wrap(", ".join(ds_ok), width=70):
        plt.text(0.05, y, line, fontsize=10); y -= 0.03
    pdf.savefig(); plt.close()

    # Por dataset
    for dname, df in dfs.items():
        S = summary_table(df).sort_values("variant")
        variants = S["variant"].tolist()

        # Test bars
        plt.figure(figsize=(10,6))
        x = np.arange(len(variants))
        plt.bar(x, S["acc_test_mean"].values, yerr=S["acc_test_std"].values, capsize=4)
        plt.xticks(x, variants, rotation=20)
        plt.ylim(0.0, 1.05); plt.grid(axis="y", alpha=0.3)
        plt.title(f"{dname} — Accuracy (test) mean ± std (n_seeds={int(S['count'].max())})")
        plt.ylabel("Accuracy test"); plt.tight_layout()
        plt.savefig(figdir / f"{dname}_test_bar.png", dpi=200)
        pdf.savefig(); plt.close()

        # Train bars
        plt.figure(figsize=(10,6))
        x = np.arange(len(variants))
        plt.bar(x, S["acc_train_mean"].values, yerr=S["acc_train_std"].values, capsize=4)
        plt.xticks(x, variants, rotation=20)
        plt.ylim(0.0, 1.05); plt.grid(axis="y", alpha=0.3)
        plt.title(f"{dname} — Accuracy (train) mean ± std")
        plt.ylabel("Accuracy train"); plt.tight_layout()
        plt.savefig(figdir / f"{dname}_train_bar.png", dpi=200)
        pdf.savefig(); plt.close()

        # Tabla resumida
        plt.figure(figsize=(10,6)); plt.axis("off")
        lines = [f"{'Variant':<14} | {'Test':<16} | {'Train':<16} | n", "-"*60]
        for _, r in S.iterrows():
            lines.append(f"{r['variant']:<14} | {fmt_pm(r['acc_test_mean'], r['acc_test_std']):<16} | {fmt_pm(r['acc_train_mean'], r['acc_train_std']):<16} | {int(r['count'])}")
        plt.text(0.01, 0.95, f"{dname} — Resumen", fontsize=14, weight="bold")
        plt.text(0.01, 0.88, "\n".join(lines), family="monospace", fontsize=10, va="top")
        plt.tight_layout(); pdf.savefig(); plt.close()

    # Tabla global final
    rows, all_cols = [], set()
    for dname, df in dfs.items():
        S = summary_table(df)
        row = {"dataset": dname}
        for _, r in S.iterrows():
            all_cols.add(r["variant"])
            row[r["variant"]] = fmt_pm(r["acc_test_mean"], r["acc_test_std"])
        rows.append(row)
    cols = ["dataset"] + sorted(all_cols)
    T = pd.DataFrame(rows)[cols].fillna("–")

    plt.figure(figsize=(11, 0.5 + 0.35*len(T))); plt.axis("off")
    plt.title("Tabla global — Accuracy test (media ± DE)", loc="left", fontsize=14)
    tab = plt.table(cellText=T.values.tolist(), colLabels=T.columns.tolist(),
                    loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.4)
    pdf.savefig(); plt.close()

print(f"[OK] PDF: {out_pdf}")
print(f"[OK] PNGs en: {figdir}")
PY
