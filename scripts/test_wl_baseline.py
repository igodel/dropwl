"""
Script: test_wl_baseline
Propósito:
    Verificar empíricamente que 1-WL (con pocas iteraciones) no separa
    el ciclo C8 del grafo disjunto 2C4 (dos ciclos de 4).

Metodología:
    - Construimos G1 = C8.
    - Construimos G2 = C4 ∪ C4 (disjunta).
    - Ejecutamos 1-WL con t en {1, 2, 3, 4, 5}.
    - Comparamos histogramas de colores finales (invariantes a permutaciones).

Resultado esperado:
    - Los histogramas resultan iguales para todos los t evaluados,
      lo cual sugiere que 1-WL no separa estos grafos (caso canónico).

Nota:
    - Esta observación sirve como baseline para luego mostrar
      cómo drop-WL sí puede distinguir con alta probabilidad bajo dropout.
"""

from typing import Dict, Any
import networkx as nx

from src.core.wl_refinement import compute_wl_colors, color_histograma

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

def construir_ciclo(n: int) -> nx.Graph:
    """
    Construye un ciclo simple C_n.

    Parámetros:
        n: número de nodos del ciclo (n >= 3).

    Retorno:
        Grafo ciclo C_n.
    """
    G = nx.cycle_graph(n)
    return G


def construir_union_disjunta_ciclos(n: int) -> nx.Graph:
    """
    Construye la unión disjunta de dos ciclos C_n.

    Parámetros:
        n: tamaño de cada ciclo (n >= 3).

    Retorno:
        Grafo que corresponde a C_n ∪ C_n.
    """
    G1 = nx.cycle_graph(n)
    G2 = nx.cycle_graph(n)
    # La unión disjunta reasigna etiquetas para evitar colisiones.
    G_union = nx.disjoint_union(G1, G2)
    return G_union


def comparar_histogramas_iguales(h1: Dict[int, int], h2: Dict[int, int]) -> bool:
    """
    Compara dos histogramas de colores por igualdad exacta.

    Parámetros:
        h1, h2: diccionarios color->conteo.

    Retorno:
        True si son exactamente iguales, False en caso contrario.
    """
    # Normaliza claves faltantes
    claves = set(h1.keys()).union(set(h2.keys()))
    for c in claves:
        if c not in h1:
            h1[c] = 0
        if c not in h2:
            h2[c] = 0
    return h1 == h2


def main() -> None:
    G_c8 = construir_ciclo(8)
    G_2c4 = construir_union_disjunta_ciclos(4)

    print("Prueba 1-WL: C8 vs 2C4")
    for t in [1, 2, 3, 4, 5]:
        colores_c8 = compute_wl_colors(G_c8, numero_de_iteraciones=t, atributo_color_inicial=None)
        colores_2c4 = compute_wl_colors(G_2c4, numero_de_iteraciones=t, atributo_color_inicial=None)

        hist_c8 = color_histograma(colores_c8)
        hist_2c4 = color_histograma(colores_2c4)

        iguales = comparar_histogramas_iguales(hist_c8, hist_2c4)

        print(f"Iteraciones WL = {t} -> histogramas iguales: {iguales}")
        print(f"  Hist C8  : {hist_c8}")
        print(f"  Hist 2C4 : {hist_2c4}")

    print("Si todos los 'iguales' son True, se confirma el baseline esperado.")


if __name__ == "__main__":
    main()
