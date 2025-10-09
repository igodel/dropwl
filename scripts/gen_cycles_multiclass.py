"""
Script: gen_cycles_multiclass.py
Tarea:
  Multiclase (3 clases): el grafo contiene al menos un ciclo de longitud k in {4,5,6}.
  (Se inyecta un C_k y opcionalmente se añaden aristas extra).
  Las clases compiten entre sí (no hay negativos "sin ciclos": cada clase tiene su k objetivo).

Salida (.npz):
  edges_list (dtype=object), labels en {0,1,2} mapeando a [C4, C5, C6], metadatos.

Uso:
  python scripts/gen_cycles_multiclass.py \
    --n 20 --n_por_clase 500 --p_extra 0.05 --seed 20250925 \
    --out data/cycles_multiclass_4_5_6_n20.npz
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import networkx as nx

def construir_ciclo(g: nx.Graph, nodos_ciclo):
    m = len(nodos_ciclo)
    for i in range(m):
        u = int(nodos_ciclo[i]); v = int(nodos_ciclo[(i+1)%m])
        g.add_edge(u,v)

def grafo_con_Ck(n, k, rng, p_extra):
    g = nx.Graph(); g.add_nodes_from(range(n))
    nodos = np.arange(n); rng.shuffle(nodos)
    construir_ciclo(g, nodos[:k].tolist())
    if p_extra > 0.0:
        for i in range(n):
            for j in range(i+1, n):
                if not g.has_edge(i,j) and rng.rand() < p_extra:
                    g.add_edge(i,j)
    return g

def grafo_a_aristas(g: nx.Graph):
    edges=[]
    for (u,v) in g.edges():
        a,b=int(u),int(v)
        if a>b: a,b=b,a
        edges.append((a,b))
    return edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n_por_clase", type=int, default=500)
    ap.add_argument("--p_extra", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, default="data/cycles_multiclass.npz")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    ks = [4,5,6]
    edges_list, labels = [], []

    # Generamos clases 0->C4, 1->C5, 2->C6
    for cls, k in enumerate(ks):
        for _ in range(args.n_por_clase):
            g = grafo_con_Ck(args.n, k, rng, args.p_extra)
            edges_list.append(grafo_a_aristas(g))
            labels.append(cls)

    idx = np.arange(len(edges_list)); rng.shuffle(idx)
    edges_list = [edges_list[i] for i in idx]
    labels = np.array([labels[i] for i in idx], dtype=int)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edges_list=np.array(edges_list, dtype=object), labels=labels,
             n=args.n, ks=np.array(ks), p_extra=args.p_extra,
             n_por_clase=args.n_por_clase, seed=args.seed)
    print("== Dataset MULTICLASS (C4/C5/C6) generado ==")
    print("Archivo:", out, "| total grafos:", len(labels), "| n:", args.n, "| ks:", ks)

if __name__ == "__main__":
    main()
