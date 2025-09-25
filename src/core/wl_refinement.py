"""
Módulo: wl_refinement
Propósito:
    Implementar el refinamiento de colores 1-WL (Weisfeiler-Leman de orden 1)
    para grafos no dirigidos simples, con estilo académico y sin azúcar sintáctica.
"""

from typing import Dict, Any, Tuple, List, Optional
import networkx as nx


def _inicializar_colores(
    grafo: nx.Graph,
    atributo_color_inicial: Optional[str]
) -> Dict[Any, int]:
    """
    Propósito:
        Asignar colores iniciales a cada nodo.
    """
    colores: Dict[Any, int] = {}
    if atributo_color_inicial is None:
        for nodo in grafo.nodes():
            colores[nodo] = 0
        return colores

    mapa_categoria_a_entero: Dict[Any, int] = {}
    proximo_entero: int = 0
    for nodo in grafo.nodes():
        valor = grafo.nodes[nodo].get(atributo_color_inicial, None)
        if valor not in mapa_categoria_a_entero:
            mapa_categoria_a_entero[valor] = proximo_entero
            proximo_entero = proximo_entero + 1
        colores[nodo] = mapa_categoria_a_entero[valor]
    return colores


def compute_wl_colors(
    grafo: nx.Graph,
    numero_de_iteraciones: int,
    atributo_color_inicial: Optional[str] = None
) -> Dict[Any, int]:
    """
    Propósito:
        Ejecutar 1-WL durante un número fijo de iteraciones.
    """
    colores_actuales: Dict[Any, int] = _inicializar_colores(grafo, atributo_color_inicial)

    if numero_de_iteraciones <= 0:
        return colores_actuales

    for _ in range(numero_de_iteraciones):
        nuevos_colores: Dict[Any, int] = {}
        firma_a_entero: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        siguiente_entero: int = 0

        for nodo in grafo.nodes():
            color_propio = colores_actuales[nodo]
            colores_vecinos_ordenados: List[int] = []
            for vecino in grafo.neighbors(nodo):
                colores_vecinos_ordenados.append(colores_actuales[vecino])
            colores_vecinos_ordenados.sort()

            firma = (color_propio, tuple(colores_vecinos_ordenados))
            if firma not in firma_a_entero:
                firma_a_entero[firma] = siguiente_entero
                siguiente_entero = siguiente_entero + 1

            nuevos_colores[nodo] = firma_a_entero[firma]

        colores_actuales = nuevos_colores

    return colores_actuales


def color_histograma(
    colores: Dict[Any, int]
) -> Dict[int, int]:
    """
    Propósito:
        Convertir un diccionario nodo->color en un histograma color->conteo.
    """
    hist: Dict[int, int] = {}
    for _, c in colores.items():
        if c not in hist:
            hist[c] = 0
        hist[c] = hist[c] + 1
    return hist
