# scripts/gen_skipcircles.py
import argparse, numpy as np, networkx as nx

def make_cycle(n):
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n): G.add_edge(i, (i+1)%n)
    return G

def make_skip_cycle(n, skip=2):
    G = make_cycle(n)
    for i in range(n):
        G.add_edge(i, (i+skip)%n)
    return G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pos = [make_skip_cycle(args.n, skip=2) for _ in range(args.n_total//2)]
    neg = [make_cycle(args.n) for _ in range(args.n_total - len(pos))]
    graphs = pos + neg
    labels = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int64)

    edges_list = np.array([np.array(list(G.edges()), dtype=np.int64) for G in graphs], dtype=object)
    np.savez_compressed(args.out, edges_list=edges_list, labels=labels, n=np.int64(args.n))
    print(f"[OK] Guardado {len(graphs)} grafos en {args.out} | n={args.n} | skip=2 vs ciclo puro")

if __name__ == "__main__":
    main()
