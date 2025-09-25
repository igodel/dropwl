"""
Módulo: dropout
Propósito:
    Implementar un operador de "node dropout" para grafos no dirigidos,
    con dos funciones:
        - sample_node_mask: genera una máscara booleana nodo->cae (True) o se mantiene (False)
        - apply_node_dropout: devuelve un nuevo grafo con los nodos caídos eliminados
Estilo:
    Código académico, explícito, sin azúcar sintáctica, con comentarios de intención.
"""

from typing import Dict, Any
import random
import networkx as nx


def sample_node_mask(
    grafo: nx.Graph,
    probabilidad_dropout: float,
    semilla: int
) -> Dict[Any, bool]:
    """
    Propósito:
        Generar una máscara que decide, para cada nodo, si "cae" (se elimina) o no.

    Parámetros:
        grafo:
            Grafo de entrada (no dirigido).
        probabilidad_dropout:
            Valor en [0, 1]. Es la probabilidad independiente de "caída" por nodo.
        semilla:
            Entero para fijar el generador pseudoaleatorio local.

    Retorno:
        Diccionario nodo -> bool, donde True indica "nodo caído" y False "nodo se mantiene".

    Notas:
        - Independencia entre nodos.
        - Para probabilidad muy pequeña, la mayoría de las máscaras tendrán 0 o 1 nodo caído.
    """
    if probabilidad_dropout < 0.0:
        probabilidad_dropout = 0.0
    if probabilidad_dropout > 1.0:
        probabilidad_dropout = 1.0

    rng = random.Random(semilla)
    mascara: Dict[Any, bool] = {}
    for nodo in grafo.nodes():
        u = rng.random()
        cae = False
        if u < probabilidad_dropout:
            cae = True
        mascara[nodo] = cae
    return mascara


def apply_node_dropout(
    grafo: nx.Graph,
    mascara_nodo_cae: Dict[Any, bool]
) -> nx.Graph:
    """
    Propósito:
        Aplicar la máscara removiendo físicamente del grafo los nodos marcados como "caídos".

    Parámetros:
        grafo:
            Grafo de entrada (no dirigido).
        mascara_nodo_cae:
            Diccionario nodo -> bool. True indica que el nodo debe eliminarse.

    Retorno:
        Un nuevo grafo de NetworkX resultante de eliminar los nodos marcados.

    Consideraciones:
        - Se crea una copia inducida por los nodos que permanecen.
        - Esta opción es más natural para WL que "poner a cero" atributos.
    """
    nodos_a_mantener = []
    for nodo in grafo.nodes():
        cae = mascara_nodo_cae.get(nodo, False)
        if cae is False:
            nodos_a_mantener.append(nodo)

    # Subgrafo inducido por nodos que permanecen
    subgrafo = grafo.subgraph(nodos_a_mantener).copy()
    return subgrafo
