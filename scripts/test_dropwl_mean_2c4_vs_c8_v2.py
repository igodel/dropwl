import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
from typing import List, Tuple
import argparse
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean


def construir_c8() -> nx.Graph:
    return nx.cycle_graph(8)

def construir_2c4() -> nx.Graph:
    return nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))

def dataset_c8_vs_2c4(n_por_clase: int, semilla: int) -> Tuple[List[nx.Graph], np.ndarray]:
    rng = np.random.RandomState(semilla)
    grafos: List[nx.Graph] = []
    etiquetas = []
    for _ in range(n_por_clase):
        grafos.append(construir_c8()); etiquetas.append(0)
    for _ in range(n_por_clase):
        grafos.append(construir_2c4()); etiquetas.append(1)
    idx = np.arange(len(grafos))
    rng.shuffle(idx)
    grafos = [grafos[k] for k in idx]
    y = np.array([etiquetas[k] for k in idx], dtype=int)
    return grafos, y

def representar_wl_baseline(g: nx.Graph, t: int, k_max: int) -> np.ndarray:
    colores = compute_wl_colors(g, numero_de_iteraciones=t, atributo_color_inicial=None)
    firma = construir_firma_histograma(colores, k_max)
    return firma

def main():
    parser = argparse.ArgumentParser(description="dropWL-mean vs WL baseline (C8 vs 2C4)")
    parser.add_argument("--n_por_clase", type=int, default=100, help="número de grafos por clase")
    parser.add_argument("--p", type=float, default=0.1, help="probabilidad de dropout por nodo")
    parser.add_argument("--R", type=int, default=100, help="número de ejecuciones dropWL")
    parser.add_argument("--t", type=int, default=3, help="iteraciones de 1-WL")
    parser.add_argument("--kmax", type=int, default=8, help="dimensión de firma por ejecución")
    parser.add_argument("--seed", type=int, default=20250925, help="semilla dataset")
    parser.add_argument("--seed_base", type=int, default=777, help="semilla base para ejecuciones")
    parser.add_argument("--solver", type=str, default="lbfgs", help="solver de LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=200, help="máximo de iteraciones LR")
    args = parser.parse_args()

    print("== Config ==")
    print(vars(args))

    t0 = time.time()
    grafos, y = dataset_c8_vs_2c4(n_por_clase=args.n_por_clase, semilla=args.seed)
    print(f"[t+{time.time()-t0:.3f}s] Dataset listo: {len(grafos)} grafos")

    # Representaciones
    t1 = time.time()
    X_wl = []
    for g in grafos:
        X_wl.append(representar_wl_baseline(g, t=args.t, k_max=args.kmax))
    X_wl = np.vstack(X_wl)
    print(f"[t+{time.time()-t1:.3f}s] WL baseline representado; X_wl shape={X_wl.shape}")

    t2 = time.time()
    X_dw = []
    for g in grafos:
        X_dw.append(
            representar_grafo_dropwl_mean(
                grafo=g, p=args.p, R=args.R, t=args.t, k_max=args.kmax, semilla_base=args.seed_base
            )
        )
    X_dw = np.vstack(X_dw)
    print(f"[t+{time.time()-t2:.3f}s] dropWL-mean representado; X_dw shape={X_dw.shape}")

    # Split y entrenamiento
    Xtr_wl, Xte_wl, ytr, yte = train_test_split(X_wl, y, test_size=0.3, random_state=42, stratify=y)
    Xtr_dw, Xte_dw, _, _ = train_test_split(X_dw, y, test_size=0.3, random_state=42, stratify=y)
    print(f"[info] Split: train={len(ytr)}, test={len(yte)}")

    t3 = time.time()
    clf_wl = LogisticRegression(max_iter=args.max_iter, solver=args.solver, n_jobs=1)
    clf_wl.fit(Xtr_wl, ytr)
    acc_wl = accuracy_score(yte, clf_wl.predict(Xte_wl))
    print(f"[t+{time.time()-t3:.3f}s] Entrenamiento WL baseline OK; acc={acc_wl:.4f}")

    t4 = time.time()
    clf_dw = LogisticRegression(max_iter=args.max_iter, solver=args.solver, n_jobs=1)
    clf_dw.fit(Xtr_dw, ytr)
    acc_dw = accuracy_score(yte, clf_dw.predict(Xte_dw))
    print(f"[t+{time.time()-t4:.3f}s] Entrenamiento dropWL-mean OK; acc={acc_dw:.4f}")

    print("== Resumen ==")
    print("WL baseline (p=0, R=1)  -> Accuracy test:", round(float(acc_wl), 4))
    print(f"dropWL-mean (p={args.p}, R={args.R}) -> Accuracy test:", round(float(acc_dw), 4))
    print(f"Tiempo total: {time.time()-t0:.3f}s")

if __name__ == "__main__":
    main()
