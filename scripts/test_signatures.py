"""
Script: test_signatures
Propósito:
    Verificar que la firma:
      1) Es invariante a permutaciones de IDs de color.
      2) Tiene dimensión fija k_max.
      3) Da el mismo resultado para C8 y 2C4 bajo 1-WL (como en Fase 1).
"""

import random
import networkx as nx
import numpy as np

from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma


def permutar_ids_de_color(colores: dict, semilla: int) -> dict:
    """
    Aplica una permutación aleatoria a los IDs de color.
    """
    rng = random.Random(semilla)
    ids = sorted(set(colores.values()))
    ids_permutados = list(ids)
    rng.shuffle(ids_permutados)
    mapa = {ids[i]: ids_permutados[i] for i in range(len(ids))}
    return {n: mapa[c] for n, c in colores.items()}


def main():
    k_max = 8
    t = 3

    # Construir grafos
    G_c8 = nx.cycle_graph(8)
    G_2c4 = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))

    # Colores WL
    colores_c8 = compute_wl_colors(G_c8, numero_de_iteraciones=t)
    colores_2c4 = compute_wl_colors(G_2c4, numero_de_iteraciones=t)

    # Firmas
    firma_c8 = construir_firma_histograma(colores_c8, k_max)
    firma_2c4 = construir_firma_histograma(colores_2c4, k_max)

    # Permutación de IDs de color
    colores_c8_perm = permutar_ids_de_color(colores_c8, semilla=42)
    firma_c8_perm = construir_firma_histograma(colores_c8_perm, k_max)

    # Reporte
    print("== Test de firmas (Fase 3) ==")
    print("Firma C8     :", firma_c8)
    print("Firma C8 perm:", firma_c8_perm)
    print("Firma 2C4    :", firma_2c4)

    # Chequeos
    def iguales(a, b, tol=1e-12):
        return np.allclose(a, b, atol=tol)

    print("Invariancia a permutación:", iguales(firma_c8, firma_c8_perm))
    print("Igualdad C8 vs 2C4 (esperado True con 1-WL):", iguales(firma_c8, firma_2c4))
    print("Dimensión fija (esperado k_max=8):", firma_c8.shape[0])


if __name__ == "__main__":
    main()
