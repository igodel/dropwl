# scripts/gen_cycles_2c4_vs_c8.py
# -*- coding: utf-8 -*-
"""
Genera un dataset balanceado para distinguir C8 vs 2C4.
- n = 8 nodos.
- Clase 0: C8 (un ciclo de longitud 8).
- Clase 1: 2C4 (dos ciclos disjuntos de longitud 4).
Guarda: edges_list (object), labels (int64), n (int)

Uso:
PYTHONPATH=. python scripts/gen_cycles_2c4_vs_c8.py \
  --n_por_clase 500 \
  --seed 20250925 \
  --augment_k 3 \
  --out data/cycles_2c4_vs_c8.npz
"""

import argparse
import numpy as np
import networkx as nx
from typing import List, Tuple

def make_C8() -> nx.Graph:
    G = nx.cycle_graph(8)  # nodos 0..7, aristas de ciclo
    return G

def make_2C4() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(8))
    # C4 en {0,1,2,3}
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    # C4 en {4,5,6,7}
    G.add_edges_from([(4,5),(5,6),(6,7),(7,4)])
    return G

def permute_graph(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Permuta etiquetas de nodos para generar variantes isomorfas."""
    n = G.number_of_nodes()
    perm = np.array(rng.permutation(n), dtype=int)
    H = nx.Graph()
    H.add_nodes_from(range(n))
    for u,v in G.edges():
        H.add_edge(int(perm[u]), int(perm[v]))
    return H

def graph_to_edge_list(G: nx.Graph) -> List[Tuple[int,int]]:
    return [(int(u), int(v)) for u,v in G.edges()]

def main():
    ap = argparse.ArgumentParser(description="Genera dataset C8 vs 2C4 (n=8).")
    ap.add_argument("--n_por_clase", type=int, default=500,
                    help="Número de grafos por clase (antes de augment).")
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--augment_k", type=int, default=3,
                    help="Número de permutaciones extra por grafo base (0=sin augment).")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Plantillas base
    base_C8  = make_C8()
    base_2C4 = make_2C4()

    edges_list = []
    labels = []

    # Generamos n_por_clase grafos por clase, cada uno con augment_k permutaciones
    # total por clase: n_por_clase*(augment_k+1)
    for _ in range(args.n_por_clase):
        # C8
        G = permute_graph(base_C8, rng)
        edges_list.append(graph_to_edge_list(G))
        labels.append(0)
        for _a in range(args.augment_k):
            H = permute_graph(base_C8, rng)
            edges_list.append(graph_to_edge_list(H))
            labels.append(0)

        # 2C4
        G = permute_graph(base_2C4, rng)
        edges_list.append(graph_to_edge_list(G))
        labels.append(1)
        for _a in range(args.augment_k):
            H = permute_graph(base_2C4, rng)
            edges_list.append(graph_to_edge_list(H))
            labels.append(1)

    edges_arr = np.array(edges_list, dtype=object)
    labels_arr = np.array(labels, dtype=np.int64)
    n = 8

    # Mezclar para evitar bloques por clase
    idx = rng.permutation(len(labels_arr))
    edges_arr = edges_arr[idx]
    labels_arr = labels_arr[idx]

    # Guardar
    out = args.out
    np.savez_compressed(out,
                        edges_list=edges_arr,
                        labels=labels_arr,
                        n=n)
    total = len(labels_arr)
    print(f"[OK] Guardado {total} grafos en {out} | n={n} | balance 50/50")
    print(f"     n_por_clase={args.n_por_clase}, augment_k={args.augment_k}")
    print(f"     Clase 0 (C8): {int((labels_arr==0).sum())} | Clase 1 (2C4): {int((labels_arr==1).sum())}")

if __name__ == "__main__":
    main()
