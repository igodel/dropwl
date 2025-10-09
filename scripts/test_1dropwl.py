# -*- coding: utf-8 -*-
# scripts/test_1dropwl.py

import time
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.core.onedropwl import wl_signature_fixed, representar_grafo_1dropwl_mean

def build_C8():
    return nx.cycle_graph(8)

def build_2C4():
    G1 = nx.cycle_graph(4)
    G2 = nx.relabel_nodes(nx.cycle_graph(4), mapping={i: i+4 for i in range(4)})
    return nx.disjoint_union(G1, G2)

def main():
    t = 1
    k_max = 8
    R = 10
    seed = 20250925
    rng = np.random.default_rng(seed)

    graphs = []
    labels = []

    for _ in range(50):
        G = build_C8()
        perm = np.array(rng.permutation(G.number_of_nodes()))
        P = nx.relabel_nodes(G, mapping={i: int(perm[i]) for i in range(G.number_of_nodes())})
        graphs.append(P)
        labels.append(0)

    for _ in range(50):
        G = build_2C4()
        perm = np.array(rng.permutation(G.number_of_nodes()))
        P = nx.relabel_nodes(G, mapping={i: int(perm[i]) for i in range(G.number_of_nodes())})
        graphs.append(P)
        labels.append(1)

    # *** CLAVE: mantener como listas Python ***
    # NO: graphs = np.array(graphs, dtype=object)
    # NO: labels = np.array(labels, dtype=np.int64)

    Gtr, Gte, ytr, yte = train_test_split(
        graphs, labels, test_size=0.3, stratify=labels, random_state=seed
    )

    # Baseline WL
    Xtr_wl = np.stack([wl_signature_fixed(G, t=t, k_max=k_max) for G in Gtr], axis=0)
    Xte_wl = np.stack([wl_signature_fixed(G, t=t, k_max=k_max) for G in Gte], axis=0)
    clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=seed)
    clf.fit(Xtr_wl, np.array(ytr, dtype=np.int64))
    acc_wl = accuracy_score(np.array(yte, dtype=np.int64), clf.predict(Xte_wl))

    # 1-dropWL
    Xtr_1d = np.stack([representar_grafo_1dropwl_mean(G, R=R, t=t, k_max=k_max, semilla_base=seed) for G in Gtr], axis=0)
    Xte_1d = np.stack([representar_grafo_1dropwl_mean(G, R=R, t=t, k_max=k_max, semilla_base=seed+1) for G in Gte], axis=0)
    clf2 = LogisticRegression(solver="lbfgs", max_iter=200, random_state=seed)
    clf2.fit(Xtr_1d, np.array(ytr, dtype=np.int64))
    acc_1d = accuracy_score(np.array(yte, dtype=np.int64), clf2.predict(Xte_1d))

    print("== Sanity-check 1-dropWL (C8 vs 2C4) ==")
    print(f"Parámetros: t={t}, k_max={k_max}, R={R}")
    print("Accuracy WL baseline     :", round(acc_wl, 4))
    print("Accuracy 1-dropWL (mean) :", round(acc_1d, 4))

if __name__ == "__main__":
    main()