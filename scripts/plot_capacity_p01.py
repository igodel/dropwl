# scripts/plot_capacity_p01.py
# Genera las 4 figuras de la pág. 9 del paper (Fig. 4) para p=0.1 usando dropWL+MLP.
# Salidas: PDF unificado + PNGs + CSVs procesados.

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def find_results(root: Path, p_target: float):
    """
    Recorre la estructura de resultados y devuelve un DataFrame con:
    n, seed, mlp_layers, mlp_hidden, acc_train, acc_test, capacity
    Filtrado para dropWL+MLP con p=p_target.
    """
    rows = []
    # patrón de ruta esperado: .../c4_n{n}/dropWL_mlp/p{p}_R.../results.csv
    pat_n = re.compile(r"c4_n(\d+)")
    pat_p = re.compile(r"p(\d+(?:\.\d+)?)_")

    for csv in root.rglob("c4_n*/dropWL_mlp/p*/results.csv"):
        # extraer n
        m_n = pat_n.search(str(csv))
        if not m_n:
            continue
        n_val = int(m_n.group(1))

        # extraer p desde el nombre de la carpeta p...
        m_p = pat_p.search(str(csv))
        if not m_p:
            continue
        p_val = float(m_p.group(1))
        if abs(p_val - p_target) > 1e-9:
            continue  # sólo p=0.1

        try:
            df = pd.read_csv(csv)
        except Exception:
            continue

        # filtrar la variante MLP
        if "variant" not in df.columns:
            continue
        df = df[df["variant"] == "dropWL+MLP"].copy()
        if df.empty:
            continue

        # columnas requeridas
        needed = ["acc_train", "acc_test", "mlp_layers", "mlp_hidden"]
        if not all(c in df.columns for c in needed):
            # intentar nombres alternativos
            alt = {
                "mlp_layers": "layers",
                "mlp_hidden": "hidden"
            }
            for k, v in alt.items():
                if k not in df.columns and v in df.columns:
                    df[k] = df[v]

        if not all(c in df.columns for c in needed):
            # no podemos usar este CSV
            continue

        # agregar n y capacity
        df["n"] = n_val
        df["capacity"] = df["mlp_layers"] * df["mlp_hidden"]

        # conservar columnas mínimas
        keep = ["n", "seed", "mlp_layers", "mlp_hidden", "capacity", "acc_train", "acc_test"]
        for k in keep:
            if k not in df.columns:
                # algunas versiones no guardaron seed; lo mapeamos a NaN
                df[k] = np.nan
        rows.append(df[keep])

    if not rows:
        raise RuntimeError("No se encontraron resultados MLP con p=%.3f bajo %s" % (p_target, root))

    ALL = pd.concat(rows, axis=0, ignore_index=True)
    # ordenar por n y capacity por prolijidad
    ALL.sort_values(["n", "capacity", "mlp_layers", "mlp_hidden"], inplace=True)
    return ALL


def panel_all_networks(axs, ALL, metric: str, ns_order):
    """
    Dibuja (a) o (b): scatter por n, con eje X=capacity, Y=metric ('acc_train' o 'acc_test').
    axs: lista de ejes (uno por n en ns_order)
    """
    for ax, n in zip(axs, ns_order):
        sub = ALL[ALL["n"] == n]
        ax.scatter(sub["capacity"], sub[metric], s=16, alpha=0.7, edgecolor="none")
        ax.set_title(f"n={n}")
        ax.set_ylim(0.45, 1.02)
        ax.set_xlim(left=0)
        ax.set_xlabel("depth × width")
        ax.set_ylabel("training accuracy" if metric == "acc_train" else "test accuracy")
        ax.grid(True, alpha=0.25)


def best_curve_by_capacity(ALL, metric: str):
    """
    Para cada n y cada capacity, toma el MÁXIMO valor del metric (sobre semillas y combinaciones que compartan capacity).
    Devuelve DF con columnas: n, capacity, best_metric
    """
    # agrupar por n y capacity: máximo del metric
    g = ALL.groupby(["n", "capacity"], as_index=False)[metric].max()
    g.rename(columns={metric: f"best_{metric}"}, inplace=True)
    return g


def panel_best_curves(ax, Gbest, metric: str, ns_order):
    """
    Dibuja (c) o (d): curvas de mejor accuracy vs capacity, una curva por n.
    """
    colors = {
        8:  "#4daf4a",
        16: "#b2df8a",
        24: "#e6ab02",
        32: "#e41a1c",
        40: "#984ea3",
        44: "#377eb8",  # por si existiera
    }
    label_metric = "training accuracy" if metric == "acc_train" else "test accuracy"

    for n in ns_order:
        sub = Gbest[Gbest["n"] == n].sort_values("capacity")
        if sub.empty:
            continue
        ax.plot(sub["capacity"], sub[f"best_{metric}"], marker="o", lw=2,
                label=f"n={n}", color=colors.get(n, None))
        ax.set_ylim(0.45, 1.02)
        ax.set_xlim(left=0)
        ax.set_xlabel("depth × width")
        ax.set_ylabel(label_metric)
        ax.grid(True, alpha=0.25)
    ax.legend(title=None, frameon=False, ncol=1)


def main():
    ap = argparse.ArgumentParser(description="Fig.4 (paper) para p fijo (p=0.1 por defecto).")
    ap.add_argument("--root", type=str, default="results/paper_c4_full",
                    help="Raíz donde están c4_n*/dropWL_mlp/p*/results.csv")
    ap.add_argument("--p", type=float, default=0.1, help="Valor de p para filtrar dropWL+MLP")
    ap.add_argument("--outdir", type=str, default="results/paper_c4_full/capacity_p01",
                    help="Directorio de salida para PDF/PNGs/CSVs")
    ap.add_argument("--ns", type=int, nargs="*", default=[8,16,24,32,40],
                    help="Orden de n a graficar (si alguno no existe, se omite)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # 1) Cargar y filtrar resultados MLP p=target
    ALL = find_results(root, p_target=args.p)
    # Si el usuario especificó un orden/ subconjunto de n, filtramos
    ns_present = sorted(ALL["n"].unique().tolist())
    ns_order = [n for n in args.ns if n in ns_present]
    if not ns_order:
        ns_order = ns_present

    # Guardar raw filtrado
    ALL.to_csv(out / "tables" / "all_points_p01.csv", index=False)

    # 2) Figuras (a) y (b): todos los puntos
    # Creamos 2 filas de subplots con len(ns_order) columnas (o 5 como en el paper si hay 5 ns)
    ncols = len(ns_order)
    if ncols == 0:
        raise RuntimeError("No hay ns disponibles para graficar.")

    # (a) training accuracy of all trained networks
    fig_a, axs_a = plt.subplots(1, ncols, figsize=(3.0*ncols, 3.0), sharey=True)
    if ncols == 1:
        axs_a = [axs_a]
    panel_all_networks(axs_a, ALL, metric="acc_train", ns_order=ns_order)
    fig_a.suptitle("(a) training accuracy of all trained networks", y=1.02, fontsize=12)
    fig_a.tight_layout()
    fig_a.savefig(out / "figures" / "panel_a_train_scatter.png", dpi=200)

    # (b) test accuracy of all trained networks
    fig_b, axs_b = plt.subplots(1, ncols, figsize=(3.0*ncols, 3.0), sharey=True)
    if ncols == 1:
        axs_b = [axs_b]
    panel_all_networks(axs_b, ALL, metric="acc_test", ns_order=ns_order)
    fig_b.suptitle("(b) test accuracy of all trained networks", y=1.02, fontsize=12)
    fig_b.tight_layout()
    fig_b.savefig(out / "figures" / "panel_b_test_scatter.png", dpi=200)

    # 3) Figuras (c) y (d): mejores curvas por capacity
    Gbest_tr = best_curve_by_capacity(ALL, metric="acc_train")
    Gbest_te = best_curve_by_capacity(ALL, metric="acc_test")
    Gbest_tr.to_csv(out / "tables" / "best_by_capacity_train.csv", index=False)
    Gbest_te.to_csv(out / "tables" / "best_by_capacity_test.csv", index=False)

    # (c) best training accuracy
    fig_c, ax_c = plt.subplots(1, 1, figsize=(5.0, 3.5))
    panel_best_curves(ax_c, Gbest_tr, metric="acc_train", ns_order=ns_order)
    fig_c.suptitle("(c) best training accuracy", y=1.02, fontsize=12)
    fig_c.tight_layout()
    fig_c.savefig(out / "figures" / "panel_c_best_train.png", dpi=200)

    # (d) best test accuracy
    fig_d, ax_d = plt.subplots(1, 1, figsize=(5.0, 3.5))
    panel_best_curves(ax_d, Gbest_te, metric="acc_test", ns_order=ns_order)
    fig_d.suptitle("(d) best test accuracy", y=1.02, fontsize=12)
    fig_d.tight_layout()
    fig_d.savefig(out / "figures" / "panel_d_best_test.png", dpi=200)

    # 4) PDF unificado (como en el paper)
    pdf_path = out / "figures" / "capacity_p01.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig_a)
        pdf.savefig(fig_b)
        pdf.savefig(fig_c)
        pdf.savefig(fig_d)

    plt.close("all")

    print("[OK] Guardado PDF:", pdf_path)
    print("[OK] PNGs en:", out / "figures")
    print("[OK] Tablas en:", out / "tables")


if __name__ == "__main__":
    main()
