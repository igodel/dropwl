import argparse, numpy as np, networkx as nx
from pathlib import Path

def make_two_cycles(n):
    """Une dos ciclos disjuntos (n debe ser par)."""
    assert n % 2 == 0
    G1 = nx.cycle_graph(n//2)
    G2 = nx.cycle_graph(n//2)
    G = nx.disjoint_union(G1, G2)
    return G

def make_one_cycle(n):
    """Un único ciclo largo de n nodos."""
    return nx.cycle_graph(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_half = args.n_total // 2

    graphs = []
    labels = []

    for _ in range(n_half):
        graphs.append(make_two_cycles(args.n))
        labels.append(0)
    for _ in range(args.n_total - n_half):
        graphs.append(make_one_cycle(args.n))
        labels.append(1)

    edges_list = np.array([np.array(list(G.edges()), dtype=np.int64) for G in graphs], dtype=object)
    labels = np.array(labels, dtype=np.int64)
    np.savez_compressed(args.out, edges_list=edges_list, labels=labels, n=np.int64(args.n))
    print(f"[OK] LIMITS-1 generado: {args.n_total} grafos (n={args.n}) → {args.out}")

if __name__ == "__main__":
    main()
