# scripts/permute_nodes_in_dataset.py
"""
Permuta aleatoriamente los IDs de nodos (0..n-1) de todos los grafos en un dataset .npz,
conservando las etiquetas y el tamaño n. Útil para chequear invariancia a permutación.
"""

import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_in', type=str, required=True)
    ap.add_argument('--data_out', type=str, required=True)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    d = np.load(args.data_in, allow_pickle=True)
    edges_list = d['edges_list']
    labels = d['labels']
    n = int(d['n']) if 'n' in d.files else None
    if n is None:
        # intenta inferir n máximo
        n = 0
        for edges in edges_list:
            if len(edges) > 0:
                mx = max(max(u,v) for (u,v) in edges)
                n = max(n, mx+1)

    perm = np.arange(n)
    rng.shuffle(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)

    edges_list_perm = []
    for edges in edges_list:
        edges = [(int(u), int(v)) for (u,v) in edges]
        edges_p = [(int(perm[u]), int(perm[v])) for (u,v) in edges]
        edges_list_perm.append(edges_p)

    np.savez(args.data_out,
             edges_list=np.array(edges_list_perm, dtype=object),
             labels=labels,
             n=np.int64(n))
    print(f"Guardado dataset permutado en: {args.data_out} (seed={args.seed})")

if __name__ == "__main__":
    main()
