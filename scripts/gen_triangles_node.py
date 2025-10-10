# scripts/gen_triangles_node.py
import argparse, numpy as np, networkx as nx
from itertools import combinations

def node_triangle_labels(G):
    tri = {v:0 for v in G.nodes()}
    for u,v,w in combinations(G.nodes(), 3):
        if G.has_edge(u,v) and G.has_edge(v,w) and G.has_edge(w,u):
            tri[u]=1; tri[v]=1; tri[w]=1
    return np.array([tri[i] for i in range(G.number_of_nodes())], dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.15)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    graphs, Y = [], []
    for _ in range(args.n_total):
        G = nx.erdos_renyi_graph(args.n, args.p, seed=int(rng.integers(1<<30)))
        y = node_triangle_labels(G)
        graphs.append(G); Y.append(y)

    edges_list = np.array([np.array(list(G.edges()), dtype=np.int64) for G in graphs], dtype=object)
    node_labels = np.stack(Y, axis=0)   # [N_graphs, n] fijo
    np.savez_compressed(args.out, edges_list=edges_list, node_labels=node_labels, n=np.int64(args.n))
    print(f"[OK] {len(graphs)} grafos en {args.out} | n={args.n} | p={args.p} | node-level labels")
if __name__ == "__main__":
    main()
