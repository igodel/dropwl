"""
Script: gen_triangles_hard.py
Tarea:
  Binaria: contiene al menos un triángulo (C3) vs no contiene C3,
  pero los negativos siguen siendo cíclicos (p. ej., C4, C5,...).

Salida (.npz):
  edges_list (dtype=object), labels (0/1), metadatos.

Uso:
  python scripts/gen_triangles_hard.py \
    --n 20 --n_por_clase 500 --p_base 0.12 --p_extra 0.05 \
    --max_reintentos 500 --seed 20250925 \
    --out data/triangles_hard_n20.npz
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import networkx as nx

def tiene_ciclo(G: nx.Graph) -> bool:
    return len(nx.cycle_basis(G)) > 0

def tiene_C3(G: nx.Graph) -> bool:
    # triángulos = cliques de tamaño 3
    # forma simple: contar triángulos con clustering (más caro) o brute-force sobre aristas
    # Aquí: simple aproximación con "nx.triangles" (cuenta triángulos por nodo)
    tri_por_nodo = nx.triangles(G)
    return sum(tri_por_nodo.values()) > 0  # cada triángulo se cuenta 3 veces

def grafo_positivo(n, rng, p_extra: float) -> nx.Graph:
    # construimos un triángulo explícito y luego añadimos aristas extra
    g = nx.Graph(); g.add_nodes_from(range(n))
    nodos = np.arange(n); rng.shuffle(nodos)
    a, b, c = map(int, nodos[:3])
    g.add_edge(a,b); g.add_edge(b,c); g.add_edge(c,a)
    if p_extra > 0.0:
        for i in range(n):
            for j in range(i+1, n):
                if not g.has_edge(i,j) and rng.rand() < p_extra:
                    g.add_edge(i,j)
    return g

def grafo_negativo_duro(n, rng, p_base: float, max_reintentos: int) -> nx.Graph:
    # muestreamos G(n,p_base) hasta que: (1) tenga ciclos, (2) NO tenga triángulos
    intentos = 0
    while intentos < max_reintentos:
        G = nx.erdos_renyi_graph(n, p_base, seed=int(rng.randint(0, 10**9)))
        if tiene_ciclo(G) and not tiene_C3(G):
            return G
        intentos += 1
    raise RuntimeError("No se pudo generar negativo duro; ajusta p_base o max_reintentos.")

def grafo_a_aristas(g: nx.Graph):
    edges = []
    for (u,v) in g.edges():
        a,b = int(u),int(v)
        if a>b: a,b=b,a
        edges.append((a,b))
    return edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n_por_clase", type=int, default=500)
    ap.add_argument("--p_base", type=float, default=0.12)
    ap.add_argument("--p_extra", type=float, default=0.05)
    ap.add_argument("--max_reintentos", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, default="data/triangles_hard.npz")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    edges_list, labels = [], []

    for _ in range(args.n_por_clase):
        gp = grafo_positivo(args.n, rng, args.p_extra)
        edges_list.append(grafo_a_aristas(gp)); labels.append(1)

    for _ in range(args.n_por_clase):
        gn = grafo_negativo_duro(args.n, rng, args.p_base, args.max_reintentos)
        edges_list.append(grafo_a_aristas(gn)); labels.append(0)

    idx = np.arange(len(edges_list)); rng.shuffle(idx)
    edges_list = [edges_list[i] for i in idx]
    labels = np.array([labels[i] for i in idx], dtype=int)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edges_list=np.array(edges_list, dtype=object), labels=labels,
             n=args.n, p_base=args.p_base, p_extra=args.p_extra,
             n_por_clase=args.n_por_clase, seed=args.seed)
    print("== Dataset TRIANGLES HARD generado ==")
    print("Archivo:", out, "| total grafos:", len(labels), "| n:", args.n)

if __name__ == "__main__":
    main()
