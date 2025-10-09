#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis consolidado estilo paper (Sección 5, C4) para p fijo (p=0.1 por defecto).

Qué hace:
- Recorre toda la carpeta raíz (--root) buscando CSVs "results.csv" de WL / dropWL-mean / dropWL+MLP.
- Filtra únicamente configuraciones de dropWL+MLP con p=valor fijado (--p).
- Consolida resultados en un master DataFrame.
- Calcula tablas:
    * Accuracy test (media±DE) por n y variante.
    * Deltas vs WL (pareadas por seed) con IC bootstrap y p-permutación.
    * Top MLP por n.
    * Tiempos promedio por variante.
- Genera figuras:
    * Boxplot de accuracy por variante.
    * Curvas de capacidad: accuracy vs #capas (L) por ancho (H), separadas por (R,D).
    * Histograma de ∆ accuracy vs WL.
    * Accuracy vs Tiempo (trade-off).
- Exporta todo a:
    out/
      results_master_p01.csv
      tables/*.csv
      figures/*.png
      report_p01.pdf (todas las láminas)

Requisitos (pip):
    pandas, numpy, matplotlib, scipy
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Matplotlib sin estilos raros, salida a PDF y PNG
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Para IC bootstrap y test de permutación
from scipy.stats import norm

# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------

def parse_variant_from_path(csv_path: Path):
    """Infere 'variant_folder' (WL / dropWL_mean / dropWL_mlp) y n a partir de la ruta."""
    parts = csv_path.parts
    # Buscar 'c4_n{n}' en la ruta
    n = None
    for p in parts:
        m = re.match(r"^c4_n(\d+)$", p)
        if m:
            n = int(m.group(1))
            break
    # variant_folder es el subdir inmediato dentro de c4_n{n}
    # Ej: .../c4_n8/WL/results.csv  -> variant_folder=WL
    #     .../c4_n8/dropWL_mean/... -> dropWL_mean
    variant_folder = None
    if n is not None:
        idx = parts.index(f"c4_n{n}")
        if idx+1 < len(parts):
            variant_folder = parts[idx+1]
    return n, variant_folder

def parse_mlp_hparams_from_folder(folder_name: str):
    """
    Carpeta MLP con patrón: p{p}_R{R}_L{layers}_H{hidden}_D{d}_{act}
    Ej.: p0.1_R50_L2_H64_D64_relu
    """
    pat = r"^p(?P<p>0\.\d+|1\.0)_R(?P<R>\d+)_L(?P<L>\d+)_H(?P<H>\d+)_D(?P<D>\d+)_(?P<act>[a-zA-Z0-9]+)$"
    m = re.match(pat, folder_name)
    if not m:
        return {}
    d = m.groupdict()
    # convertir numéricos
    out = {
        "mlp_p": float(d["p"]),
        "mlp_R": int(d["R"]),
        "mlp_layers": int(d["L"]),
        "mlp_hidden": int(d["H"]),
        "mlp_d": int(d["D"]),
        "mlp_act": d["act"]
    }
    return out

def load_all_results(root: Path):
    """Carga todos los results.csv y concatena, añadiendo columnas inferidas."""
    rows = []
    for csv in root.rglob("results.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        n, variant_folder = parse_variant_from_path(csv)
        if n is None or variant_folder is None:
            continue
        df["n"] = n
        df["variant_folder"] = variant_folder

        # Normalizar "variant" a etiquetas finales: WL, dropWL-mean, dropWL+MLP
        # Si la columna 'variant' ya viene correcta, respetarla; si no, mapear desde folder.
        if "variant" in df.columns and df["variant"].notna().all():
            pass
        else:
            mapping = {"WL": "WL", "dropWL_mean": "dropWL-mean", "dropWL_mlp": "dropWL+MLP"}
            df["variant"] = df["variant_folder"].map(mapping)

        # En MLP, inferir hiperparámetros desde la carpeta (padre de results.csv)
        if variant_folder == "dropWL_mlp":
            mlp_folder = csv.parent.name  # nombre p0.1_R50_L2_H64_D64_relu
            hp = parse_mlp_hparams_from_folder(mlp_folder)
            for k, v in hp.items():
                df[k] = v

        rows.append(df)

    if not rows:
        raise SystemExit(f"[ERR] No se encontraron CSVs en {root}")
    return pd.concat(rows, axis=0, ignore_index=True)

def paired_delta_vs_wl(df_level):
    """
    Calcula deltas pareados (acc_variant - acc_WL) por seed.
    df_level contiene columnas: seed, variant, acc_test.
    Retorna DataFrame con columnas: seed, variant, delta
    (descarta WL en el resultado, deja solo variantes vs WL).
    """
    out = []
    seeds = sorted(df_level["seed"].unique())
    for s in seeds:
        sub = df_level[df_level["seed"] == s]
        if "WL" not in sub["variant"].values:
            continue
        acc_wl = float(sub[sub["variant"] == "WL"]["acc_test"].mean())
        for v in sub["variant"].unique():
            if v == "WL": 
                continue
            acc_v = float(sub[sub["variant"] == v]["acc_test"].mean())
            out.append({"seed": s, "variant": v, "delta": acc_v - acc_wl})
    if not out:
        return pd.DataFrame(columns=["seed","variant","delta"])
    return pd.DataFrame(out)

def bootstrap_ci(deltas, B=10000, alpha=0.05, rng=None):
    """IC bootstrap percentil para la media de deltas (pareadas)."""
    if rng is None:
        rng = np.random.default_rng(123)
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return np.nan, np.nan
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boots.append(deltas[idx].mean())
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1 - alpha/2))
    return float(lo), float(hi)

def paired_permutation_pvalue(deltas, B=10000, rng=None):
    """
    Test de permutación apareado tipo 'sign-flip':
    H0: E[delta]=0 vs H1: E[delta] != 0.
    """
    if rng is None:
        rng = np.random.default_rng(456)
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return np.nan
    obs = deltas.mean()
    count = 0
    for _ in range(B):
        signs = rng.choice([-1, 1], size=n)
        perm = (deltas * signs).mean()
        if abs(perm) >= abs(obs):
            count += 1
    return (count + 1)/(B + 1)

# ---------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Análisis consolidado p=fijo (estilo paper) para C4")
    ap.add_argument("--root", type=str, required=True, help="Raíz de resultados (outroot del orquestador)")
    ap.add_argument("--p", type=float, default=0.1, help="Valor fijo de p para filtrar MLP (ej. 0.1)")
    ap.add_argument("--out", type=str, required=True, help="Carpeta de salida donde guardar tablas/figuras/PDF")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # 1) Cargar todo
    DF = load_all_results(root)

    # Guardar master completo (p=0.1 filtrado solo en MLP; WL y dropWL-mean se mantienen)
    DF.to_csv(out / "results_master_raw.csv", index=False)

    # 2) Filtrar MLP por p fijo
    mask_mlp = (DF["variant"] == "dropWL+MLP")
    DF_mlp = DF[mask_mlp].copy()
    if "mlp_p" in DF_mlp.columns:
        DF_mlp = DF_mlp[np.isclose(DF_mlp["mlp_p"], args.p)].copy()
    else:
        # Si no hay columna mlp_p, no hay MLP parametrizado; dejar vacío
        DF_mlp = DF_mlp.head(0).copy()

    # WL y dropWL-mean (no dependen de mlp_p)
    DF_wl = DF[DF["variant"] == "WL"].copy()
    DF_mean = DF[DF["variant"] == "dropWL-mean"].copy()

    # Unir para análisis conjunto p=fijo
    cols_common = list(set(DF.columns) | {"mlp_p","mlp_R","mlp_layers","mlp_hidden","mlp_d","mlp_act"})
    DFp = pd.concat([DF_wl[DF_wl.columns], DF_mean[DF_mean.columns], DF_mlp[DF_mlp.columns]], ignore_index=True)
    DFp.to_csv(out / "results_master_p_fixed.csv", index=False)

    # 3) Tabla: Accuracy (media ± DE) por n y variante
    acc_by_n_var = DFp.groupby(["n","variant"])["acc_test"].agg(["mean","std","count"]).reset_index()
    acc_by_n_var.to_csv(out / "tables" / "acc_by_n_variant.csv", index=False)

    # 4) Deltas pareados vs WL (por n)
    rows = []
    for n, dfn in DFp.groupby("n"):
        deltas = paired_delta_vs_wl(dfn[["seed","variant","acc_test"]].copy())
        if deltas.empty:
            continue
        for v, dfg in deltas.groupby("variant"):
            di = dfg["delta"].to_numpy()
            lo, hi = bootstrap_ci(di, B=10000, alpha=0.05)
            pval = paired_permutation_pvalue(di, B=10000)
            rows.append({
                "n": n,
                "compare": f"{v} vs WL",
                "delta_mean": float(di.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "p_perm_two_sided": pval,
                "n_pairs": len(di)
            })
    paired_table = pd.DataFrame(rows)
    paired_table.to_csv(out / "tables" / "paired_vs_WL_by_n.csv", index=False)

    # 5) Mejor MLP por n (si existe)
    best_rows = []
    if (DFp["variant"]=="dropWL+MLP").any():
        mlp_cols = ["n","seed","acc_test","mlp_p","mlp_R","mlp_layers","mlp_hidden","mlp_d","mlp_act","t_total_s"]
        have = [c for c in mlp_cols if c in DFp.columns]
        for n, dfn in DFp[DFp["variant"]=="dropWL+MLP"][have].groupby("n"):
            # mejor por media sobre seeds dentro de una misma (R,L,H,D,act)
            key_cols = ["mlp_R","mlp_layers","mlp_hidden","mlp_d","mlp_act"]
            grp = (dfn.groupby(key_cols)["acc_test"].mean().reset_index()
                     .sort_values("acc_test", ascending=False))
            if not grp.empty:
                top = grp.iloc[0]
                cfg = dfn
                for k in key_cols:
                    cfg = cfg[cfg[k] == top[k]]
                acc_mean = cfg["acc_test"].mean()
                acc_std  = cfg["acc_test"].std(ddof=1)
                t_mean   = cfg["t_total_s"].mean() if "t_total_s" in cfg else np.nan
                row = {"n": n, "acc_mean": acc_mean, "acc_std": acc_std, "t_mean": t_mean}
                for k in key_cols:
                    row[k] = top[k]
                best_rows.append(row)
    best_mlp = pd.DataFrame(best_rows)
    if not best_mlp.empty:
        best_mlp.to_csv(out / "tables" / "best_mlp_by_n.csv", index=False)

    # 6) Tiempos promedio por variante (global)
    time_by_var = DFp.groupby("variant")["t_total_s"].agg(["mean","std","count"]).reset_index()
    time_by_var.to_csv(out / "tables" / "time_by_variant.csv", index=False)

    # ---------------------------------------------------------
    # FIGURAS
    # ---------------------------------------------------------
    pdf_path = out / "report_p_fixed.pdf"
    with PdfPages(pdf_path) as pdf:

        # 6.1 Boxplot accuracy por variante (todos los n)
        plt.figure(figsize=(6,4))
        order = ["WL", "dropWL-mean", "dropWL+MLP"]
        data = [DFp[DFp["variant"]==v]["acc_test"].values for v in order if (DFp["variant"]==v).any()]
        labels = [v for v in order if (DFp["variant"]==v).any()]
        plt.boxplot(data, labels=labels)
        plt.ylabel("Accuracy (test)")
        plt.title(f"Distribución de Accuracy por variante (p={args.p})")
        plt.tight_layout()
        pdf.savefig(); plt.savefig(out / "figures" / "box_acc_by_variant.png", dpi=200)
        plt.close()

        # 6.2 Accuracy medio vs n por variante
        piv = acc_by_n_var.pivot(index="n", columns="variant", values="mean")
        plt.figure(figsize=(6,4))
        for col in piv.columns:
            plt.plot(piv.index, piv[col], marker="o", label=col)
        plt.xlabel("n (nodos)")
        plt.ylabel("Accuracy medio (test)")
        plt.title(f"Accuracy vs tamaño del grafo (p={args.p})")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(); plt.savefig(out / "figures" / "acc_vs_n.png", dpi=200)
        plt.close()

        # 6.3 Histograma de ∆ vs WL (por variante, juntando n)
        deltas_all = []
        for n, dfn in DFp.groupby("n"):
            deltas = paired_delta_vs_wl(dfn[["seed","variant","acc_test"]])
            if deltas.empty: 
                continue
            deltas["n"] = n
            deltas_all.append(deltas)
        if deltas_all:
            DEL = pd.concat(deltas_all, axis=0, ignore_index=True)
            plt.figure(figsize=(6,4))
            for v in DEL["variant"].unique():
                vals = DEL[DEL["variant"]==v]["delta"].values
                plt.hist(vals, bins=10, alpha=0.6, label=v)
            plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
            plt.xlabel("Δ Accuracy vs WL (pareada por seed)")
            plt.ylabel("Frecuencia")
            plt.title(f"Distribución de mejoras vs WL (p={args.p})")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(); plt.savefig(out / "figures" / "delta_hist.png", dpi=200)
            plt.close()

        # 6.4 Trade-off Accuracy vs Tiempo (media global por variante)
        if not time_by_var.empty:
            # usar media de accuracy global por variante
            acc_global = DFp.groupby("variant")["acc_test"].mean()
            t_global = time_by_var.set_index("variant")["mean"]
            keep = acc_global.index.intersection(t_global.index)
            plt.figure(figsize=(6,4))
            plt.scatter(t_global[keep], acc_global[keep])
            for v in keep:
                plt.annotate(v, (t_global[v], acc_global[v]), xytext=(5,5), textcoords="offset points")
            plt.xlabel("Tiempo total medio por run (s)")
            plt.ylabel("Accuracy medio (test)")
            plt.title(f"Trade-off accuracy/tiempo (p={args.p})")
            plt.tight_layout()
            pdf.savefig(); plt.savefig(out / "figures" / "acc_vs_time.png", dpi=200)
            plt.close()

        # 6.5 Curvas de capacidad: accuracy vs capas (L), curvas por H, subplots por (R,D)
        mlp_ok = (DFp["variant"]=="dropWL+MLP") & ("mlp_layers" in DFp.columns)
        if mlp_ok.any():
            # Filtrar MLP p=fijo
            M = DFp[mlp_ok].copy()
            if "mlp_p" in M.columns:
                M = M[np.isclose(M["mlp_p"], args.p)].copy()

            # Agregamos por (n, seed, R, D, L, H): media sobre nada (cada fila es un run),
            # luego promedio sobre seeds para curvas.
            # Haremos un grid por (R,D) y promediaremos sobre seeds y n separados.
            # Figura por cada n para claridad.
            for n, df_n in M.groupby("n"):
                R_values = sorted(df_n["mlp_R"].unique())
                D_values = sorted(df_n["mlp_d"].unique())
                if len(R_values)*len(D_values) == 0:
                    continue
                fig, axes = plt.subplots(len(R_values), len(D_values), figsize=(4*len(D_values), 3*len(R_values)), squeeze=False)
                for i, Rv in enumerate(R_values):
                    for j, Dv in enumerate(D_values):
                        ax = axes[i][j]
                        sub = df_n[(df_n["mlp_R"]==Rv) & (df_n["mlp_d"]==Dv)]
                        if sub.empty:
                            ax.axis('off'); continue
                        # pivot: L en eje x, líneas por H
                        grp = sub.groupby(["mlp_layers","mlp_hidden"])["acc_test"].mean().reset_index()
                        for Hval, dH in grp.groupby("mlp_hidden"):
                            dH = dH.sort_values("mlp_layers")
                            ax.plot(dH["mlp_layers"], dH["acc_test"], marker="o", label=f"H={Hval}")
                        ax.set_title(f"n={n} | R={Rv} | D={Dv}")
                        ax.set_xlabel("Capas (L)")
                        ax.set_ylabel("Accuracy (test)")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                plt.tight_layout()
                pdf.savefig(); 
                fig.savefig(out / "figures" / f"capacity_L_curves_n{n}.png", dpi=200)
                plt.close(fig)

    # Guardar también CSV maestros pos-análisis
    DFp.to_csv(out / "results_master_p01.csv", index=False)
    print(f"[OK] Análisis completo. Salida en: {out}")
    print(f"[OK] Tablas: {out/'tables'}")
    print(f"[OK] Figuras: {out/'figures'}")
    print(f"[OK] Reporte PDF: {out/'report_p_fixed.pdf'}")

if __name__ == "__main__":
    main()
