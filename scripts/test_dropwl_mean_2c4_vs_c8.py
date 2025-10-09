"""
Script: test_dropwl_mean_2c4_vs_c8
Propósito:
    Mostrar empíricamente que:
      - WL determinista (p=0, R=1) no separa C8 y 2C4 (accuracy ~ azar).
      - dropWL-mean (p pequeño, R moderado) sí separa con alta accuracy.

Metodología:
    - Dataset: 100 grafos C8 (y=0) y 100 grafos 2C4 (y=1).
    - Representaciones:
        * WL baseline: t=3, p=0, R=1
        * dropWL-mean: t=3, p=0.1 (ó 1/n), R=100
      k_max = 8 (tamaño máximo de clases razonable para estos grafos).
    - Clasificador: LogisticRegression (scikit-learn), split 70/30 estratificado.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import List, Tuple
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
    G1 = nx.cycle_graph(4)
    G2 = nx.cycle_graph(4)
    G = nx.disjoint_union(G1, G2)
    return G


def dataset_c8_vs_2c4(n_por_clase: int, semilla: int) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Construye un dataset con n_por_clase grafos de cada tipo.
    Retorna lista de grafos y vector de etiquetas (0 para C8, 1 para 2C4).
    """
    rng = np.random.RandomState(semilla)
    grafos: List[nx.Graph] = []
    etiquetas = []

    i = 0
    while i < n_por_clase:
        grafos.append(construir_c8())
        etiquetas.append(0)
        i = i + 1

    j = 0
    while j < n_por_clase:
        grafos.append(construir_2c4())
        etiquetas.append(1)
        j = j + 1

    # Permutar orden para no sesgar splits
    indices = np.arange(len(grafos))
    rng.shuffle(indices)
    grafos = [grafos[k] for k in indices]
    etiquetas = np.array([etiquetas[k] for k in indices], dtype=int)
    return grafos, etiquetas


def representar_wl_baseline(g: nx.Graph, t: int, k_max: int) -> np.ndarray:
    """
    Representación 'WL sin dropout': p=0, R=1 (equivale a una sola ejecución sobre el grafo original).
    """
    colores = compute_wl_colors(g, numero_de_iteraciones=t, atributo_color_inicial=None)
    firma = construir_firma_histograma(colores, k_max)
    return firma


def main() -> None:
    # Configuración experimental
    n_por_clase = 100
    semilla_dataset = 20250925
    t = 3
    k_max = 8

    # Hiperparámetros dropWL-mean
    p = 0.1      # también puedes probar p = 1/8 = 0.125
    R = 100
    semilla_base = 777

    # Construir dataset
    grafos, y = dataset_c8_vs_2c4(n_por_clase=n_por_clase, semilla=semilla_dataset)

    # Representaciones: WL baseline y dropWL-mean
    X_wl = []
    X_dropwl = []

    for g in grafos:
        # baseline WL (p=0, R=1)
        x_wl = representar_wl_baseline(g, t=t, k_max=k_max)
        X_wl.append(x_wl)

        # dropWL-mean (p>0, R>=1)
        x_dw = representar_grafo_dropwl_mean(
            grafo=g, p=p, R=R, t=t, k_max=k_max, semilla_base=semilla_base
        )
        X_dropwl.append(x_dw)

    X_wl = np.vstack(X_wl)
    X_dropwl = np.vstack(X_dropwl)

    # Split estratificado
    Xtr_wl, Xte_wl, ytr, yte = train_test_split(X_wl, y, test_size=0.3, random_state=42, stratify=y)
    Xtr_dw, Xte_dw, _, _ = train_test_split(X_dropwl, y, test_size=0.3, random_state=42, stratify=y)

    # Clasificador lineal (sin trucos)
    clf_wl = LogisticRegression(max_iter=1000)
    clf_dw = LogisticRegression(max_iter=1000)

    clf_wl.fit(Xtr_wl, ytr)
    clf_dw.fit(Xtr_dw, ytr)

    ypred_wl = clf_wl.predict(Xte_wl)
    ypred_dw = clf_dw.predict(Xte_dw)

    acc_wl = accuracy_score(yte, ypred_wl)
    acc_dw = accuracy_score(yte, ypred_dw)

    print("== Experimento dropWL-mean (C8 vs 2C4) ==")
    print("WL baseline (p=0, R=1)  -> Accuracy test:", round(float(acc_wl), 4))
    print(f"dropWL-mean (p={p}, R={R}) -> Accuracy test:", round(float(acc_dw), 4))
    print("Nota: esperamos acc_wl ≈ 0.5 (no separa) y acc_dw significativamente mayor (separa).")


if __name__ == "__main__":
    main()
