"""
Script: test_dropout_statistics
Propósito:
    Verificar empíricamente que la frecuencia observada de "1-dropouts" (cae exactamente 1 nodo)
    coincide con la probabilidad teórica para un grafo simple con n nodos.

Metodología:
    - Usamos un ciclo C_n, que es un grafo regular y simple para controlar n.
    - Fijamos una probabilidad p pequeña (ej.: p = 0.1 o p = 1/n, 2/n).
    - Ejecutamos R repeticiones, cada una con una semilla distinta.
    - Contamos en cuántas repeticiones "cae exactamente 1 nodo".
    - Comparamos:
        observada ≈ (n * p * (1 - p)^(n-1)) dentro de un margen razonable (ley de los grandes números).

Parámetros sugeridos:
    n = 20, p en {1/n, 2/n, 0.1}, R = 10000 (si el tiempo lo permite; para una corrida corta usar R = 2000).

Criterio de aceptación:
    La frecuencia observada debe aproximar la probabilidad teórica
    (la diferencia relativa debería ser pequeña, y decrecer con R).
"""

from typing import Tuple
import math
import networkx as nx

from src.core.dropout import sample_node_mask, apply_node_dropout


def prob_teorica_un_drop(n: int, p: float) -> float:
    """
    Probabilidad de que caiga exactamente 1 nodo (modelo Bernoulli independiente):
        P = n * p * (1 - p)^(n - 1)
    """
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    valor = n * p * ((1.0 - p) ** (n - 1))
    return valor


def contar_un_dropouts_en_R(
    grafo: nx.Graph,
    p: float,
    R: int,
    semilla_base: int
) -> Tuple[int, float]:
    """
    Ejecuta R repeticiones de dropout y cuenta cuántas veces cae exactamente 1 nodo.

    Retorno:
        (conteo_uno, frecuencia_observada)
    """
    n = grafo.number_of_nodes()
    conteo_uno = 0
    rep = 0
    while rep < R:
        semilla = semilla_base + rep
        mascara = sample_node_mask(grafo, p, semilla)
        caidos = 0
        for nodo in grafo.nodes():
            if mascara[nodo] is True:
                caidos = caidos + 1
        if caidos == 1:
            conteo_uno = conteo_uno + 1
        rep = rep + 1

    frecuencia_observada = 0.0
    if R > 0:
        frecuencia_observada = float(conteo_uno) / float(R)
    return conteo_uno, frecuencia_observada


def main() -> None:
    # Configuración base
    n = 20
    p = 1.0 / float(n)  # Sugerencia: p = 1/n enfatiza 1-dropouts
    R = 5000            # Aumenta a 10000 si quieres mayor precisión
    semilla_base = 20250923

    # Construimos un ciclo para fijar n
    G = nx.cycle_graph(n)

    # Cálculo teórico
    p_teorica = prob_teorica_un_drop(n, p)

    # Estimación empírica
    conteo_uno, freq_obs = contar_un_dropouts_en_R(G, p, R, semilla_base)

    # Reporte
    print("== Test estadístico de 1-dropouts ==")
    print("n nodos            :", n)
    print("p                  :", p)
    print("R repeticiones     :", R)
    print("Prob. teórica (1)  :", p_teorica)
    print("Conteo observado   :", conteo_uno)
    print("Frecuencia obser.  :", freq_obs)
    # Error absoluto y relativo
    error_abs = abs(freq_obs - p_teorica)
    error_rel = 0.0
    if p_teorica > 0.0:
        error_rel = error_abs / p_teorica
    print("Error absoluto     :", error_abs)
    print("Error relativo     :", error_rel)


if __name__ == "__main__":
    main()
