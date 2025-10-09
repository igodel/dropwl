# src/core/onedropwl.py
"""
Módulo: onedropwl
Propósito:
    Implementar 1-dropWL con eliminación de UNA arista por ejecución.
    Opcionalmente, repetir R veces (cada vez UNA arista distinta) y promediar.

Definición acordada:
    1-dropWL (R=1): tomar G, eliminar UNA arista elegida al azar (u.a.r.) y
    luego computar WL-t y su firma (histograma normalizado de colores, dimensión fija k_max).

    1edge-dropWL-mean (R>1): repetir el procedimiento anterior R veces (cada vez
    UNA arista al azar) y promediar los R vectores finales (promedio aritmético).

Notas:
    - Si |E(G)| = 0, no hay aristas que eliminar; se retorna la firma WL de G.
    - Para reproducibilidad, se usa numpy.random.default_rng(seed).
    - Firma WL robusta a la variante de compute_wl_colors (t_iter, t o posicional).
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import networkx as nx

# Import WL (robusto a firmas distintas)
from src.core.wl_refinement import compute_wl_colors, color_histograma


def _wl_signature_fixed(G: nx.Graph, t: int, k_max: int) -> np.ndarray:
    """
    Devuelve la firma WL (vector de longitud k_max) tras t iteraciones,
    normalizada por frecuencia, tolerante a distintas firmas de compute_wl_colors.
    """
    # Llamada robusta: t_iter -> t -> posicional
    try:
        colores = compute_wl_colors(G, t_iter=t)
    except TypeError:
        try:
            colores = compute_wl_colors(G, t=t)
        except TypeError:
            colores = compute_wl_colors(G, t)

    hist = color_histograma(colores)  # dict color_id -> conteo
    total = sum(hist.values()) if hist else 1
    vec = np.zeros(k_max, dtype=np.float64)
    for cid, cnt in hist.items():
        if 0 <= cid < k_max:
            vec[cid] = float(cnt) / float(total)
    return vec


def _remove_one_random_edge(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """
    Devuelve una copia de G con EXACTAMENTE UNA arista eliminada (u.a.r.).
    Si G no tiene aristas, retorna una copia sin cambios.
    """
    H = G.copy()
    edges = list(H.edges())
    if len(edges) == 0:
        # No hay nada que eliminar
        return H
    idx = rng.integers(low=0, high=len(edges))
    e = edges[idx]
    H.remove_edge(*e)
    return H


def onedropwl_one_edge(G: nx.Graph, t: int, k_max: int, seed: Optional[int] = None) -> np.ndarray:
    """
    1-dropWL (R=1): elimina UNA arista al azar y calcula la firma WL.
    """
    rng = np.random.default_rng(seed)
    H = _remove_one_random_edge(G, rng)
    vec = _wl_signature_fixed(H, t=t, k_max=k_max)
    return vec.astype(np.float32)


def onedropwl_one_edge_mean(
    G: nx.Graph,
    t: int,
    k_max: int,
    R: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    1edge-dropWL-mean (R>1): repetir R veces el borrado de UNA arista (cada vez nueva muestra),
    y promediar los R vectores WL resultantes.
    """
    rng = np.random.default_rng(seed)
    reps = []
    for r in range(R):
        H = _remove_one_random_edge(G, rng)
        v = _wl_signature_fixed(H, t=t, k_max=k_max)
        reps.append(v)
    if len(reps) == 0:
        # No debería ocurrir (R>=1), pero por robustez:
        return _wl_signature_fixed(G, t=t, k_max=k_max).astype(np.float32)
    vec = np.mean(np.stack(reps, axis=0), axis=0)
    return vec.astype(np.float32)


def represent_batch_onedrop_one(
    graphs: List[nx.Graph],
    t: int,
    k_max: int,
    seed_base: int
) -> np.ndarray:
    """
    Representa un lote de grafos con 1-dropWL (R=1).
    Usa seeds distintos por grafo para reproducibilidad.
    """
    X = []
    for i, G in enumerate(graphs):
        vec = onedropwl_one_edge(G, t=t, k_max=k_max, seed=seed_base + i)
        X.append(vec)
    return np.stack(X, axis=0)


def represent_batch_onedrop_mean(
    graphs: List[nx.Graph],
    t: int,
    k_max: int,
    R: int,
    seed_base: int
) -> np.ndarray:
    """
    Representa un lote de grafos con 1edge-dropWL-mean (R>1).
    """
    X = []
    for i, G in enumerate(graphs):
        vec = onedropwl_one_edge_mean(G, t=t, k_max=k_max, R=R, seed=seed_base + i)
        X.append(vec)
    return np.stack(X, axis=0)