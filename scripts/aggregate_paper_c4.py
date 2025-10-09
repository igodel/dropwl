# scripts/aggregate_paper_c4.py
import argparse
from pathlib import Path
import pandas as pd

def load_all_results(root: Path):
    rows = []
    for csv in root.rglob("results.csv"):
        # inferir n / variante / config desde la ruta
        parts = csv.relative_to(root).parts
        # Ejemplo: c4_n8/WL/results.csv
        n_part = parts[0]  # c4_n8
        n = int(n_part.split('_')[2][1:])  # "n8" -> 8
        variant = parts[1]                 # "WL" / "dropWL_mean" / "dropWL_mlp"
        # Cargar y enriquecer
        df = pd.read_csv(csv)
        df["n"] = n
        df["variant_folder"] = variant
        rows.append(df)
    if not rows:
        raise SystemExit(f"[ERR] No se encontraron CSVs en {root}")
    return pd.concat(rows, axis=0, ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Agregador resultados C4 (paper-like)")
    ap.add_argument("--root", type=str, required=True, help="Carpeta raíz (outroot de run_paper_c4_all.py)")
    args = ap.parse_args()

    root = Path(args.root)
    df = load_all_results(root)
    df.to_csv(root / "results_master.csv", index=False)
    print(f"[OK] master: {root/'results_master.csv'}")

    # Resumen por (n, variant)
    # Normalizamos nombres: "dropWL_mean" -> "dropWL-mean"; "dropWL_mlp" -> "dropWL+MLP"
    mapping = {"WL":"WL", "dropWL_mean":"dropWL-mean", "dropWL_mlp":"dropWL+MLP"}
    df["variant"] = df["variant"].where(df["variant"].isin(mapping.values()),
                                        df["variant_folder"].map(mapping))
    # tabla por n
    table_n = df.groupby(["n","variant"])["acc_test"].agg(["mean","std","count"]).reset_index()
    table_n.to_csv(root / "tables_acc_by_n_variant.csv", index=False)
    print(f"[table] {root/'tables_acc_by_n_variant.csv'}")

    # Mejor MLP por n (si existe)
    if (df["variant"]=="dropWL+MLP").any():
        cols = ["n","seed","acc_test","mlp_p","mlp_R","mlp_layers","mlp_hidden","mlp_d","mlp_act"]
        mlp = df[df["variant"]=="dropWL+MLP"][cols].copy()
        best = mlp.groupby("n").apply(lambda x: x.sort_values("acc_test", ascending=False).head(1)).reset_index(drop=True)
        best.to_csv(root / "tables_best_mlp_by_n.csv", index=False)
        print(f"[table] {root/'tables_best_mlp_by_n.csv'}")

    # Gráficos simples (si hay matplotlib)
    try:
        import matplotlib.pyplot as plt
        # Acc vs n por variante
        piv = table_n.pivot(index="n", columns="variant", values="mean")
        plt.figure(figsize=(6,4))
        for col in piv.columns:
            plt.plot(piv.index, piv[col], marker="o", label=col)
        plt.xlabel("n (nodos)")
        plt.ylabel("Accuracy test (media)")
        plt.title("C4: Accuracy vs tamaño del grafo")
        plt.legend()
        (root / "figures").mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(root / "figures" / "acc_vs_n.png", dpi=200)
        print(f"[fig] {root/'figures'/'acc_vs_n.png'}")
    except Exception as e:
        print("[WARN] No se generaron figuras:", repr(e))

if __name__ == "__main__":
    main()