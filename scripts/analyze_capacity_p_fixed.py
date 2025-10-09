#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analiza capacidad (depth x width) para p fijo (=0.1 por defecto) y genera
un informe tipo paper con 4 paneles:
 (a) train accuracy (todos los runs)
 (b) test accuracy  (todos los runs)
 (c) best train accuracy por capacidad
 (d) best test  accuracy por capacidad

Asume que bajo --root hay subcarpetas por n (p.ej., c4_n8, c4_n16, ...)
y dentro de cada una hay subcarpetas de variantes/arquitecturas con un
archivo results.csv (como los que generan tus scripts exp_c4_compare/run_paper*).

Columnas esperadas (mínimo):
  - variant (string)
  - seed (int)
  - acc_train, acc_test (float)
  - mlp_layers (int), mlp_hidden (int)  -> definen "capacity"
  - n (int)                              -> si no existe, se infiere del path
  - mlp_p (float)                        -> si existe, se filtra por --p

Robustez:
  - Si no encuentra mlp_p/dw_p, no filtra por p (y advierte).
  - Si no encuentra n, intenta inferir n del path (patrón c4_nXX).
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- utilidades ----------

def infer_n_from_path(path: Path) -> Optional[int]:
    """
    Intenta inferir n del path usando patrón 'c4_nXX' o '_nXX_'.
    Devuelve int o None si no se puede.
    """
    s = str(path)
    # patrones comunes: 'c4_n8', 'c4_n16', '..._n40_...'
    m = re.search(r'[_/-]n(\d+)[_/-]', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r'c4_n(\d+)', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def load_all_results(root: Path) -> pd.DataFrame:
    """
    Recorre recursivamente --root y concatena todos los results.csv
    que encuentre. Devuelve un DataFrame con una columna extra 'src' (origen).
    """
    rows: List[pd.DataFrame] = []
    for csv in root.rglob("results.csv"):
        try:
            df = pd.read_csv(csv)
            df["src"] = str(csv)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] No pude leer {csv}: {e}", file=sys.stderr)
    if not rows:
        raise FileNotFoundError(f"No encontré results.csv bajo {root}")
    DF = pd.concat(rows, axis=0, ignore_index=True)

    # si no hay 'n', intentar inferirlo del path
    if "n" not in DF.columns:
        n_inferred = []
        for s in DF["src"]:
            n_inferred.append(infer_n_from_path(Path(s)))
        DF["n"] = n_inferred
        if DF["n"].isna().any():
            print("[WARN] No se pudo inferir 'n' de algunas rutas; revisa tus paths.", file=sys.stderr)
            # por sanidad, elimina filas sin n
            DF = DF.dropna(subset=["n"])
        DF["n"] = DF["n"].astype(int)

    return DF


def ensure_capacity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas mlp_layers y mlp_hidden; calcula 'capacity' = layers*hidden.
    Si faltan, aborta con mensaje claro.
    """
    missing = [c for c in ["mlp_layers", "mlp_hidden"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas para capacidad: {missing}. "
            f"Necesito mlp_layers y mlp_hidden en tus CSV."
        )
    df = df.copy()
    df["capacity"] = (df["mlp_layers"].astype(int) * df["mlp_hidden"].astype(int)).astype(int)
    return df


def filter_variant_and_p(df: pd.DataFrame, variant: str, p_fixed: Optional[float]) -> pd.DataFrame:
    """
    Filtra por variant y por p si hay columna mlp_p o dw_p.
    Si no existe ninguna columna de p y p_fixed no es None, advierte.
    """
    df = df[df["variant"] == variant].copy()
    if df.empty:
        raise ValueError(f"No hay filas para variant='{variant}'. Revisa tus resultados.")

    # detectar columna de p
    p_col = None
    for cand in ["mlp_p", "dw_p"]:
        if cand in df.columns:
            p_col = cand
            break

    if p_fixed is not None:
        if p_col is None:
            print("[WARN] No hay columna de probabilidad p (mlp_p/dw_p); no puedo filtrar por p. "
                  "Asumo que todos los runs son p fijo en este conjunto.",
                  file=sys.stderr)
        else:
            df = df[np.isclose(df[p_col].astype(float), float(p_fixed))].copy()
            if df.empty:
                raise ValueError(
                    f"No quedan filas tras filtrar por {p_col} == {p_fixed}. "
                    f"Revisa que tus corridas incluyan ese p."
                )

    return df


def panel_scatter_all(ax, df_n: pd.DataFrame, ycol: str, title: str):
    """
    Dibuja un scatter de todos los runs (capacity en X, ycol en Y) para un n fijo.
    """
    x = df_n["capacity"].values
    y = df_n[ycol].values
    ax.scatter(x, y, s=16, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Capacidad (layers × hidden)")
    ax.set_ylabel(("Accuracy train" if ycol == "acc_train" else "Accuracy test"))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.02)


def panel_curve_best(ax, df_n: pd.DataFrame, ycol: str, title: str):
    """
    Dibuja la curva de mejor accuracy por capacidad (max sobre runs con igual capacidad).
    """
    best = df_n.groupby("capacity")[ycol].max().reset_index().sort_values("capacity")
    ax.plot(best["capacity"].values, best[ycol].values, marker="o", ms=4)
    ax.set_title(title)
    ax.set_xlabel("Capacidad (layers × hidden)")
    ax.set_ylabel(("Mejor accuracy train" if ycol == "acc_train" else "Mejor accuracy test"))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.02)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Capacidad (p fijo) — report (a–d)")
    ap.add_argument("--root", type=str, required=True,
                    help="Carpeta raíz con subcarpetas c4_nXX/.../results.csv")
    ap.add_argument("--out", type=str, default="report_p_fixed.pdf", help="PDF de salida")
    ap.add_argument("--variant", type=str, default="dropWL+MLP", help="Variante a analizar (p.ej., dropWL+MLP)")
    ap.add_argument("--p", type=float, default=0.1, help="Probabilidad de dropout a filtrar (si está en CSV)")
    ap.add_argument("--ns", type=int, nargs="+", default=[8, 16, 24, 32, 40],
                    help="Lista de n a incluir (si existen)")
    args = ap.parse_args()

    root = Path(args.root)
    out_pdf = Path(args.out)

    # 1) cargar todo
    DF = load_all_results(root)

    # 2) filtrar variant + p fijo
    DF = filter_variant_and_p(DF, args.variant, args.p)

    # 3) asegurar capacidad y sanear
    DF = ensure_capacity_columns(DF)
    # sanear columnas métricas
    for col in ["acc_train", "acc_test"]:
        if col not in DF.columns:
            raise ValueError(f"Falta la columna {col} en tus CSV.")
        DF[col] = DF[col].astype(float)

    # 4) preparar figura multi-panel (a–d)
    # Haremos 4 filas (a,b,c,d), y tantas columnas como ns encontradas
    ns_present = [n for n in args.ns if (n in set(DF["n"].tolist()))]
    if not ns_present:
        raise ValueError("No hay valores de 'n' de la lista --ns presentes en los resultados.")

    ncols = len(ns_present)
    fig, axs = plt.subplots(4, ncols, figsize=(4.2 * ncols, 12.0), dpi=120, squeeze=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # títulos de filas (a–d)
    row_titles = [
        "(a) Training accuracy of all trained networks",
        "(b) Test accuracy of all trained networks",
        "(c) Best training accuracy",
        "(d) Best test accuracy"
    ]

    # 5) poblar paneles
    for j, nval in enumerate(ns_present):
        df_n = DF[DF["n"] == nval].copy()
        # (a) scatter de training
        panel_scatter_all(axs[0, j], df_n, "acc_train", f"n={nval}")
        # (b) scatter de test
        panel_scatter_all(axs[1, j], df_n, "acc_test", f"n={nval}")
        # (c) curva best training
        panel_curve_best(axs[2, j], df_n, "acc_train", f"n={nval}")
        # (d) curva best test
        panel_curve_best(axs[3, j], df_n, "acc_test", f"n={nval}")

    # poner títulos de filas en el margen izquierdo
    for i in range(4):
        axs[i, 0].set_ylabel(axs[i, 0].get_ylabel() + "\n" + row_titles[i], fontsize=10)

    # 6) guardar PDF y PNGs auxiliares
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[OK] PDF: {out_pdf}")

    # además, exportar paneles individuales
    base = out_pdf.with_suffix("")
    tags = ["train_all", "test_all", "train_best", "test_best"]
    for i, tag in enumerate(tags):
        subfig = plt.figure(figsize=(4.2 * ncols, 3.0), dpi=120)
        grid = plt.GridSpec(1, ncols)
        for j, nval in enumerate(ns_present):
            ax = subfig.add_subplot(grid[0, j])
            df_n = DF[DF["n"] == nval].copy()
            if i == 0:
                panel_scatter_all(ax, df_n, "acc_train", f"n={nval}")
            elif i == 1:
                panel_scatter_all(ax, df_n, "acc_test", f"n={nval}")
            elif i == 2:
                panel_curve_best(ax, df_n, "acc_train", f"n={nval}")
            else:
                panel_curve_best(ax, df_n, "acc_test", f"n={nval}")
        png_path = Path(f"{base}_{tag}.png")
        subfig.tight_layout()
        subfig.savefig(png_path, bbox_inches="tight")
        plt.close(subfig)
        print(f"[OK] PNG: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
