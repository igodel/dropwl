"""
Módulo: dropwl_runner
Propósito:
    Implementar el motor de dropWL-mean:
        - Ejecuta R ejecuciones independientes:
            * node dropout con probabilidad p
            * 1-WL por t iteraciones
            * firma histograma (k_max)
        - Agrega por media las R firmas para obtener la representación del grafo.

Entradas:
    - Grafo G (networkx.Graph)
    - p: probabilidad de dropout por nodo (float en [0,1])
    - R: número de ejecuciones (int)
    - t: iteraciones de 1-WL (int)
    - k_max: dimensión de la firma por ejecución (int)
    - semilla_base: entero para reproducibilidad (cada ejecución usa semilla_base + i)

Salida:
    - Vector numpy de dimensión (k_max,), correspondiente a la media de las R firmas.
"""

from typing import Optional
import numpy as np
import networkx as nx

from src.core.dropout import sample_node_mask, apply_node_dropout
from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma


def representar_grafo_dropwl_mean(
    grafo: nx.Graph,
    p: float,
    R: int,
    t: int,
    k_max: int,
    semilla_base: int = 12345
) -> np.ndarray:
    """
    Propósito:
        Obtener la representación dropWL-mean de un grafo.

    Retorno:
        Vector numpy de dimensión (k_max,), correspondiente a la media
        de las R firmas por ejecución.
    """
    if R <= 0:
        R = 1
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0

    acumulador = np.zeros((k_max,), dtype=float)
    i = 0
    while i < R:
        semilla = semilla_base + i

        # 1) muestrear máscara y construir subgrafo inducido
        mascara = sample_node_mask(grafo, p, semilla)
        subgrafo = apply_node_dropout(grafo, mascara)

        # 2) ejecutar 1-WL t iteraciones
        colores = compute_wl_colors(subgrafo, numero_de_iteraciones=t, atributo_color_inicial=None)

        # 3) construir firma histograma k_max
        firma = construir_firma_histograma(colores, k_max)

        # 4) acumular
        acumulador = acumulador + firma
        i = i + 1

    representacion_media = acumulador / float(R)
    return representacion_media
