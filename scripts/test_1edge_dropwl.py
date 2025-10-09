# scripts/test_1edge_dropwl.py
"""
Test mínimo de 1-edge-dropWL:
    - Comprueba que se elimina exactamente una arista.
    - Imprime una firma WL de ejemplo.
"""

import networkx as nx
import numpy as np
from src.core.onedropwl import onedropwl_one_edge, onedropwl_one_edge_mean, _remove_one_random_edge, _wl_signature_fixed

def main():
    # Grafo de prueba: ciclo C8 (8 nodos, 8 aristas)
    G = nx.cycle_graph(8)
    print("C8 original: |E| =", G.number_of_edges())

    # Comprobar que _remove_one_random_edge elimina exactamente una
    H = _remove_one_random_edge(G, np.random.default_rng(123))
    print("Tras remove_one_random_edge: |E| =", H.number_of_edges(), "(esperado 7)")

    # Firma 1-dropWL (R=1)
    vec1 = onedropwl_one_edge(G, t=3, k_max=8, seed=777)
    print("Firma 1-edge-dropWL (len=8):", vec1)

    # Firma 1edge-dropWL-mean (R=10)
    vec10 = onedropwl_one_edge_mean(G, t=3, k_max=8, R=10, seed=777)
    print("Firma mean de 10 repeticiones (len=8):", vec10)

if __name__ == "__main__":
    main()
