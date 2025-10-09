# scripts/test_verify_1drop_distinction.py
"""
Test controlado:
Verifica que 1-drop (1-edge-drop) elimine exactamente una arista y
compare su poder de distinción con WL en el caso 2C4 vs C8.
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.core.onedropwl import onedropwl_one_edge
from src.core.wl_refinement import compute_wl_colors, color_histograma

# --------------------------
# 1. Definiciones auxiliares
# --------------------------

def wl_signature(G, t=3, kmax=20):
    """Firma WL-t clásica (robusta a variantes de compute_wl_colors)."""
    try:
        colores = compute_wl_colors(G, t_iter=t)
    except TypeError:
        try:
            colores = compute_wl_colors(G, t=t)
        except TypeError:
            colores = compute_wl_colors(G, t)
    hist = color_histograma(colores)
    total = sum(hist.values()) if hist else 1
    vec = np.zeros(kmax)
    for cid, cnt in hist.items():
        if cid < kmax:
            vec[cid] = cnt / total
    return vec


def generate_cycles_dataset(n_per_class=10, seed=123):
    """Genera pares de grafos 2C4 (dos ciclos de 4) y C8 (un ciclo de 8)."""
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    for i in range(n_per_class):
        G1 = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))  # dos C4 disjuntos
        G2 = nx.cycle_graph(8)  # un solo C8
        graphs += [G1, G2]
        labels += [0, 1]
    return graphs, np.array(labels, dtype=int)


def count_edges_removed(G):
    """Verifica cuántas aristas se eliminaron en una operación de 1-drop."""
    import copy
    nE_before = G.number_of_edges()
    H = copy.deepcopy(G)
    from src.core.onedropwl import _remove_one_random_edge
    H2 = _remove_one_random_edge(H, np.random.default_rng(42))
    nE_after = H2.number_of_edges()
    return nE_before - nE_after


# --------------------------
# 2. Verificación del "1-drop"
# --------------------------
def verify_1drop_behavior():
    G = nx.cycle_graph(8)
    diff = count_edges_removed(G)
    print(f"[check] C8 original: {G.number_of_edges()} aristas -> eliminadas {diff} (esperado = 1)")

    # Comprobamos varias ejecuciones
    from src.core.onedropwl import onedropwl_one_edge
    vecs = [onedropwl_one_edge(G, t=3, k_max=8, seed=s) for s in range(5)]
    diffs = np.std(np.stack(vecs, axis=0), axis=0)
    print(f"[check] Varianza media entre 5 ejecuciones (debería ser >0): {diffs.mean():.4e}")


# --------------------------
# 3. Comparación con WL
# --------------------------
def compare_wl_vs_1drop():
    graphs, labels = generate_cycles_dataset(n_per_class=10)
    Gtr, Gte, ytr, yte = train_test_split(graphs, labels, test_size=0.3, stratify=labels, random_state=42)

    # Representaciones WL
    Xtr_wl = np.stack([wl_signature(G, t=3, kmax=20) for G in Gtr])
    Xte_wl = np.stack([wl_signature(G, t=3, kmax=20) for G in Gte])

    # Representaciones 1-drop
    Xtr_drop = np.stack([onedropwl_one_edge(G, t=3, k_max=20, seed=i) for i, G in enumerate(Gtr)])
    Xte_drop = np.stack([onedropwl_one_edge(G, t=3, k_max=20, seed=i+100) for i, G in enumerate(Gte)])

    sc = StandardScaler()
    Xtr_wl, Xte_wl = sc.fit_transform(Xtr_wl), sc.transform(Xte_wl)
    Xtr_drop, Xte_drop = sc.fit_transform(Xtr_drop), sc.transform(Xte_drop)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr_wl, ytr)
    acc_wl = accuracy_score(yte, clf.predict(Xte_wl))

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr_drop, ytr)
    acc_drop = accuracy_score(yte, clf.predict(Xte_drop))

    print(f"[result] Accuracy WL:      {acc_wl:.3f}")
    print(f"[result] Accuracy 1-drop:  {acc_drop:.3f}")
    print("[note] WL debería ≈0.5 (no distingue); 1-drop debería >0.5 (rompe simetría).")


# --------------------------
# 4. Main
# --------------------------
def main():
    print("== Verificación estructural de 1-drop ==")
    verify_1drop_behavior()
    print("\n== Comparación WL vs 1-drop en 2C4 vs C8 ==")
    compare_wl_vs_1drop()

if __name__ == "__main__":
    main()
