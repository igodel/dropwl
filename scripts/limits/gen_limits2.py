import argparse, numpy as np, networkx as nx

def make_cubic_circulant(n, step=2):
    """3-regular circulant: conecta cada i con (i±1)%n y (i±step)%n"""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in [(i+1)%n, (i-1)%n, (i+step)%n]:
            G.add_edge(i, j)
    return G

def make_bipartite_regular(n):
    """3-regular bipartito con grupos de n/2 y conexiones cruzadas fijas"""
    assert n % 2 == 0
    G = nx.Graph()
    A = range(n//2)
    B = range(n//2, n)
    for a in A:
        # conectar cada a con 3 nodos distintos en B (mod n/2)
        for k in range(3):
            G.add_edge(a, n//2 + (a + k) % (n//2))
    return G

def main():
    import pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_half = args.n_total // 2
    graphs, labels = [], []

    for _ in range(n_half):
        graphs.append(make_cubic_circulant(args.n, step=2))
        labels.append(0)
    for _ in range(args.n_total - n_half):
        graphs.append(make_bipartite_regular(args.n))
        labels.append(1)

    edges_list = np.array([np.array(list(G.edges()), dtype=np.int64) for G in graphs], dtype=object)
    labels = np.array(labels, dtype=np.int64)
    np.savez_compressed(args.out, edges_list=edges_list, labels=labels, n=np.int64(args.n))
    print(f"[OK] LIMITS-2 generado: {args.n_total} grafos (n={args.n}) → {args.out}")

if __name__ == "__main__":
    main()
