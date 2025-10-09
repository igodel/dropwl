"""
Módulo: graph_signatures
Propósito:
    Implementar la firma base de dropWL: un histograma normalizado,
    ordenado y con padding, de dimensión fija k_max.

Entrada:
    - Diccionario nodo -> color (resultado de 1-WL).
    - Valor k_max (entero positivo).

Salida:
    - Vector numpy de dimensión k_max, que representa la partición de colores.

Características:
    - Invariante a permutaciones de nodos.
    - Invariante a renombramientos de colores.
    - De dimensión fija (relleno con ceros si hay menos clases).
"""

from typing import Dict, Any, List
import numpy as np


def construir_firma_histograma(
    colores: Dict[Any, int],
    k_max: int
) -> np.ndarray:
    """
    Propósito:
        Construir la firma invariante de dropWL basada en histograma.

    Parámetros:
        colores:
            Diccionario nodo -> color entero final (producido por 1-WL).
        k_max:
            Dimensión fija del vector de salida.

    Retorno:
        Vector numpy de dimensión (k_max,).
    """
    # Paso 1: contar tamaños de cada clase de color
    conteos_por_color = {}
    for _, c in colores.items():
        if c not in conteos_por_color:
            conteos_por_color[c] = 0
        conteos_por_color[c] = conteos_por_color[c] + 1

    conteos: List[int] = list(conteos_por_color.values())

    # Paso 2: ordenar descendente
    conteos_ordenados = sorted(conteos, reverse=True)

    # Paso 3: normalizar (suma = 1.0)
    total = float(sum(conteos_ordenados))
    normalizados = []
    if total > 0.0:
        for v in conteos_ordenados:
            normalizados.append(float(v) / total)
    else:
        normalizados.append(0.0)

    # Paso 4: padding o recorte
    if len(normalizados) > k_max:
        normalizados = normalizados[0:k_max]
    while len(normalizados) < k_max:
        normalizados.append(0.0)

    # Paso 5: devolver como vector numpy
    firma = np.array(normalizados, dtype=float)
    return firma
