#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

DATASETS = ["C4_n8","C4_n16","C4_n24","C4_n32","C4_n40","C4_n44",
            "LIMITS1","LIMITS2","SKIP32","LCC32","TRI32"]
VARIANTS = ["WL","1drop-LOG","1drop-MLP"]

def main():
    ap = argparse.ArgumentParser(description="Revisa progreso de Fase 1 (WL / 1drop-LOG / 1drop-MLP)")
    ap.add_argument("--root", type=str, default="results/fase1_simple_all")
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="Lista de seeds esperados (ej. 20250925 20250926 ...)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] No existe la carpeta {root}")
        return

    total_expected = len(DATASETS) * len(VARIANTS) * len(args.seeds)
    total_found = 0
    datasets_done = 0

    print(f"[INFO] root={root}")
    print(f"[INFO] datasets={len(DATASETS)}, variants={len(VARIANTS)}, seeds={len(args.seeds)}")
    print("-"*72)

    for ds in DATASETS:
        csv = root / ds / "results.csv"
        if not csv.exists():
            print(f"[....] {ds:<8} -> falta results.csv")
            continue

        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"[FAIL] {ds:<8} -> no pude leer results.csv: {e}")
            continue

        ok = True
        per_variant_found = 0
        for v in VARIANTS:
            sub = df[df["variant"] == v]
            # 1 fila por seed
            seeds_found = sorted(sub["seed"].unique().tolist())
            total_found += len(seeds_found)

            # chequear seeds exactos
            missing = sorted(set(args.seeds) - set(seeds_found))
            extra   = sorted(set(seeds_found) - set(args.seeds))

            if missing or extra or len(seeds_found) != len(args.seeds):
                ok = False
                print(f"[WARN] {ds:<8} {v:<10} -> {len(seeds_found)}/{len(args.seeds)} seeds "
                      f"(faltan={missing}, extras={extra})")
            else:
                per_variant_found += 1

        if ok and per_variant_found == len(VARIANTS):
            datasets_done += 1
            # resumen corto (media ± DE) si está completo
            s = df.groupby("variant")["acc_test"].agg(["mean","std","count"]).round(4)
            print(f"[OK]   {ds:<8} COMPLETO")
            print(s.to_string())
        else:
            print(f"[TODO] {ds:<8} incompleto")

        print("-"*72)

    print(f"[RESUMEN] datasets completos: {datasets_done}/{len(DATASETS)}")
    print(f"[RESUMEN] corridas encontradas: {total_found}/{total_expected} "
          f"(datasets*variants*seeds)")

if __name__ == "__main__":
    main()