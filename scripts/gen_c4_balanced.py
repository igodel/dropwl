# scripts/gen_c4_balanced.py
import argparse, numpy as np, networkx as nx
from pathlib import Path

def find_4cycles(G):
    cyc4 = set()
    for u in G:
        Nu = set(G[u])
        for v in Nu:
            if v <= u: continue
            Nv = set(G[v])
            for w in Nv - {u}:
                if w <= v: continue
                Nw = set(G[w])
                for x in Nw - {v}:
                    if x <= w or x == u: continue
                    if G.has_edge(x, u):
                        cyc4.add(tuple(sorted([u,v,w,x])))
    return list(cyc4)

def gen_one(n, p, must_have, rng):
    G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1<<30)))
    cyc = find_4cycles(G)
    if must_have:
        if not cyc and n>=4:
            nodes = rng.choice(n, size=4, replace=False)
            a,b,c,d = nodes
            G.add_edge(int(a),int(b)); G.add_edge(int(b),int(c))
            G.add_edge(int(c),int(d)); G.add_edge(int(d),int(a))
    else:
        while find_4cycles(G):
            u,v,w,x = find_4cycles(G)[0]
            if G.has_edge(u,v): G.remove_edge(u,v)
            else: break
    return G

def to_edges_list(graphs):
    out = []
    for G in graphs:
        out.append(np.array(list(G.edges()), dtype=np.int64))
    return np.array(out, dtype=object)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    npos = args.n_total//2; nneg = args.n_total - npos
    pos = [gen_one(args.n, args.p, True, rng)  for _ in range(npos)]
    neg = [gen_one(args.n, args.p, False, rng) for _ in range(nneg)]
    graphs = pos + neg
    labels = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int64)

    edges_list = to_edges_list(graphs)
    np.savez_compressed(args.out, edges_list=edges_list, labels=labels, n=np.int64(args.n))
    print(f"[OK] Guardado {len(graphs)} grafos en {args.out} | n={args.n} | p={args.p} | balanceado 50/50")

if __name__ == "__main__":
    main()
