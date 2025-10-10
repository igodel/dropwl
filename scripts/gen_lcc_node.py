# scripts/gen_lcc_node.py
import argparse, numpy as np, networkx as nx

def node_lcc_labels(G):
    comps = list(nx.connected_components(G))
    if not comps:
        return np.zeros(G.number_of_nodes(), dtype=np.int64)
    largest = max(comps, key=len)
    y = np.zeros(G.number_of_nodes(), dtype=np.int64)
    for v in largest: y[v]=1
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    graphs, Y = [], []
    for _ in range(args.n_total):
        G = nx.erdos_renyi_graph(args.n, args.p, seed=int(rng.integers(1<<30)))
        y = node_lcc_labels(G)
        graphs.append(G); Y.append(y)

    edges_list = np.array([np.array(list(G.edges()), dtype=np.int64) for G in graphs], dtype=object)
    node_labels = np.stack(Y, axis=0)
    np.savez_compressed(args.out, edges_list=edges_list, node_labels=node_labels, n=np.int64(args.n))
    print(f"[OK] {len(graphs)} grafos en {args.out} | n={args.n} | p={args.p} | node-level LCC")
if __name__ == "__main__":
    main()
