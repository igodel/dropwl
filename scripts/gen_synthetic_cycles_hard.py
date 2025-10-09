"""
Script: gen_synthetic_cycles_hard.py
Propósito:
  Dataset binario "contiene C_k" (sí/no) con negativos DIFICILES:
  Positivos: inyecta C_k (y opcionalmente aristas extra).
  Negativos: G(n,p_base) que NO tiene C_k pero SÍ tiene algún ciclo.

Salida (.npz): edges_list, labels, n, k, p_base, p_extra, n_por_clase, seed.

Uso:
  python scripts/gen_synthetic_cycles_hard.py \
    --n 20 --k 6 --n_por_clase 500 \
    --p_base 0.12 --p_extra 0.05 \
    --max_reintentos 500 --seed 20250925 \
    --out data/cycles_hard_n20_k6.npz
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import networkx as nx
from typing import List, Tuple

def construir_ciclo(g: nx.Graph, nodos_ciclo: list) -> None:
    m = len(nodos_ciclo)
    for i in range(m):
        u = int(nodos_ciclo[i]); v = int(nodos_ciclo[(i+1) % m])
        g.add_edge(u, v)

def contiene_Ck(G: nx.Graph, k: int, limite: int = 10000) -> bool:
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v in G.edges():
        DG.add_edge(u, v); DG.add_edge(v, u)
    c = 0
    for cyc in nx.simple_cycles(DG):
        if len(cyc) == k:
            return True
        c += 1
        if c >= limite:
            break
    return False

def tiene_ciclo(G: nx.Graph) -> bool:
    return len(nx.cycle_basis(G)) > 0

def grafo_positivo(n: int, k: int, rng: np.random.RandomState, p_extra: float) -> nx.Graph:
    g = nx.Graph(); g.add_nodes_from(range(n))
    nodos = np.arange(n); rng.shuffle(nodos)
    construir_ciclo(g, nodos[:k].tolist())
    if p_extra > 0.0:
        for i in range(n):
            for j in range(i+1, n):
                if not g.has_edge(i, j) and rng.rand() < p_extra:
                    g.add_edge(i, j)
    return g

def grafo_negativo_duro(n: int, k: int, rng: np.random.RandomState, p_base: float, max_reintentos: int) -> nx.Graph:
    intentos = 0
    while intentos < max_reintentos:
        G = nx.erdos_renyi_graph(n, p_base, seed=int(rng.randint(0, 10**9)))
        if tiene_ciclo(G) and not contiene_Ck(G, k):
            return G
        intentos += 1
    raise RuntimeError("No se pudo generar negativo duro. Ajusta p_base o aumenta max_reintentos.")

def grafo_a_aristas(g: nx.Graph) -> list[tuple[int,int]]:
    edges = []
    for (u, v) in g.edges():
        a, b = int(u), int(v)
        if a > b:
            a, b = b, a
        edges.append((a, b))
    return edges

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_por_clase", type=int, default=500)
    ap.add_argument("--p_base", type=float, default=0.12)
    ap.add_argument("--p_extra", type=float, default=0.05)
    ap.add_argument("--max_reintentos", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, default="data/cycles_hard.npz")
    args = ap.parse_args()

    if args.k < 3:
        raise ValueError("k debe ser >= 3")

    rng = np.random.RandomState(args.seed)
    edges_list, labels = [], []

    for _ in range(args.n_por_clase):
        gp = grafo_positivo(args.n, args.k, rng, args.p_extra)
        edges_list.append(grafo_a_aristas(gp))
        labels.append(1)

    for _ in range(args.n_por_clase):
        gn = grafo_negativo_duro(args.n, args.k, rng, args.p_base, args.max_reintentos)
        edges_list.append(grafo_a_aristas(gn))
        labels.append(0)

    idx = np.arange(len(edges_list)); rng.shuffle(idx)
    edges_list = [edges_list[i] for i in idx]
    labels = np.array([labels[i] for i in idx], dtype=int)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edges_list=np.array(edges_list, dtype=object), labels=labels,
             n=args.n, k=args.k, p_base=args.p_base, p_extra=args.p_extra,
             n_por_clase=args.n_por_clase, seed=args.seed)
    print("== Dataset HARD generado ==")
    print("Archivo:", out, "| total grafos:", len(labels), "| n:", args.n, "| k:", args.k,
          "| p_base:", args.p_base, "| p_extra:", args.p_extra)

if __name__ == "__main__":
    main()
