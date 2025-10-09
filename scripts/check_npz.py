#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check mínimo para verificar lectura de .npz (dtype=object) y reconstrucción de un grafo.
"""

import argparse
import numpy as np
import networkx as nx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    edges_list = data["edges_list"]   # dtype=object
    labels = data["labels"].astype(np.int64)
    n = int(data["n"])
    print(f"[OK] leídos {len(edges_list)} grafos | n={n}")

    # Reconstruir primer grafo
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges_list[0])
    print(f"[info] G0: |V|={G.number_of_nodes()} |E|={G.number_of_edges()} |label0={labels[0]}")

if __name__ == "__main__":
    main()
