"""
Script: gen_synthetic_cycles.py
Propósito:
    Generar un dataset binario "contiene C_k" (sí/no) para n nodos:
      - Positivos: grafo con un ciclo simple de longitud exacta k (inyectado).
      - Negativos: árbol aleatorio (acíclico, no contiene C_k).
    Opción de agregar aristas extra aleatorias en positivos para aumentar dificultad.

Salida:
    Un archivo .npz con:
        - edges_list: lista de listas de aristas [(u,v), ...] por grafo (dtype=object)
        - labels: np.ndarray de 0/1
        - n, k, p_extra, n_por_clase, seed (metadatos)

Uso:
    python scripts/gen_synthetic_cycles.py --n 20 --k 6 --n_por_clase 500 --p_extra 0.05 --seed 20250925 --out data/cycles_n20_k6.npz
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List, Tuple
import numpy as np
import networkx as nx


def construir_ciclo_en_grafo(g: nx.Graph, nodos_ciclo: List[int]) -> None:
    """
    Agrega al grafo 'g' las aristas que forman un ciclo simple sobre 'nodos_ciclo' en ese orden.
    """
    m = len(nodos_ciclo)
    i = 0
    while i < m:
        u = nodos_ciclo[i]
        v = nodos_ciclo[(i + 1) % m]
        g.add_edge(u, v)
        i = i + 1


def grafo_positivo_con_Ck(n: int, k: int, rng: np.random.RandomState, p_extra: float) -> nx.Graph:
    """
    Construye un grafo con n nodos que contiene al menos un C_k.
    Procedimiento:
      - Crea grafo vacío con n nodos.
      - Elige k nodos al azar y les agrega un ciclo simple (C_k).
      - Agrega aristas extra con probabilidad p_extra entre pares (para mayor dificultad).
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    nodos = np.arange(n); rng.shuffle(nodos)
    ciclo = list(nodos[:k])
    construir_ciclo_en_grafo(g, ciclo)

    if p_extra > 0.0:
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                if rng.rand() < p_extra:
                    g.add_edge(int(i), int(j))
                j = j + 1
            i = i + 1

    return g


def grafo_negativo_sin_ciclos(n: int, rng: np.random.RandomState) -> nx.Graph:
    """
    Construye un grafo acíclico (árbol aleatorio con n nodos).
    - Garantiza que NO contiene C_k (ni ningún ciclo).
    """
    # networkx.random_tree devuelve un árbol (n-1 aristas), acíclico por definición.
    T = nx.random_tree(n, seed=int(rng.randint(0, 10**9)))
    # Asegurar tipo Graph (no MultiGraph), nodos como int consecutivos
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(T.edges())
    return g


def grafo_a_lista_aristas(g: nx.Graph) -> List[Tuple[int, int]]:
    """
    Convierte un grafo a una lista de aristas con u<v (aristas sin dirección).
    """
    edges = []
    for (u, v) in g.edges():
        a = int(u); b = int(v)
        if a < b:
            edges.append((a, b))
        else:
            edges.append((b, a))
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador de dataset sintético: detección de C_k (sí/no)")
    parser.add_argument("--n", type=int, default=20, help="número de nodos por grafo")
    parser.add_argument("--k", type=int, default=6, help="longitud del ciclo objetivo (k >= 3)")
    parser.add_argument("--n_por_clase", type=int, default=500, help="número de grafos por clase")
    parser.add_argument("--p_extra", type=float, default=0.05, help="probabilidad de arista extra en positivos")
    parser.add_argument("--seed", type=int, default=20250925, help="semilla RNG")
    parser.add_argument("--out", type=str, default="data/cycles_dataset.npz", help="ruta de salida .npz")
    args = parser.parse_args()

    if args.k < 3:
        raise ValueError("k debe ser >= 3")

    rng = np.random.RandomState(args.seed)
    edges_list = []
    labels = []

    # Positivos (1): contienen C_k
    i = 0
    while i < args.n_por_clase:
        g_pos = grafo_positivo_con_Ck(n=args.n, k=args.k, rng=rng, p_extra=args.p_extra)
        edges_list.append(grafo_a_lista_aristas(g_pos))
        labels.append(1)
        i = i + 1

    # Negativos (0): acíclicos (árbol)
    j = 0
    while j < args.n_por_clase:
        g_neg = grafo_negativo_sin_ciclos(n=args.n, rng=rng)
        edges_list.append(grafo_a_lista_aristas(g_neg))
        labels.append(0)
        j = j + 1

    # Barajar (mezclar clases)
    idx = np.arange(len(edges_list))
    rng.shuffle(idx)
    edges_list = [edges_list[t] for t in idx]
    labels = np.array([labels[t] for t in idx], dtype=int)

    # Guardar
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, edges_list=np.array(edges_list, dtype=object), labels=labels,
             n=args.n, k=args.k, p_extra=args.p_extra, n_por_clase=args.n_por_clase, seed=args.seed)

    print("== Dataset generado ==")
    print("Archivo:", str(out_path))
    print("Grafos totales:", len(labels))
    print("n nodos:", args.n, "| k ciclo:", args.k, "| p_extra:", args.p_extra)
    print("positivos (1):", args.n_por_clase, "| negativos (0):", args.n_por_clase)


if __name__ == "__main__":
    main()
