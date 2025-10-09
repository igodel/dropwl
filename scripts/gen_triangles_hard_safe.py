#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación robusta de dataset TRIÁNGULOS (negativos "duros") para n grandes.

No modifica tu generador original. Este script:
  - Clase positiva (label=1): grafo con triángulos (ER con p_pos >= p_base + p_extra)
  - Clase negativa "dura" (label=0): grafo sin triángulos o casi sin triángulos,
    construido de forma CONSTRUCTIVA:
      * Punto de partida: grafo bipartito aleatorio G_{n/2,n/2}(q) (no tiene triángulos).
      * Si quieres ajustar densidad hacia un objetivo ~p_base, se añade/saca aristas
        SIN crear triángulos: sólo se agregan pares (u,v) con N(u)∩N(v)=∅.

Esto evita el rechazo infinito y escala bien para n=80 (y más).
"""

import argparse
from pathlib import Path
import numpy as np
import networkx as nx

def triangles_count(G: nx.Graph) -> int:
    tri_per_node = nx.triangles(G)
    return sum(tri_per_node.values()) // 3

def random_bipartite_no_triangles(n: int, q: float, rng: np.random.RandomState) -> nx.Graph:
    """Crea un grafo bipartito completo por partición (no hay triángulos)."""
    assert n >= 2
    left = list(range(n//2))
    right = list(range(n//2, n))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in left:
        for v in right:
            if rng.rand() < q:
                G.add_edge(u, v)
    return G

def can_add_without_triangle(G: nx.Graph, u: int, v: int) -> bool:
    if G.has_edge(u, v):
        return False
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    return len(Nu.intersection(Nv)) == 0

def adjust_density_without_triangles(G: nx.Graph, m_target: int, rng: np.random.RandomState, max_trials: int = 200000):
    """
    Ajusta el nº de aristas hacia m_target SIN crear triángulos:
      - Si m < m_target: intenta agregar aristas (u,v) con N(u)∩N(v)=∅
      - Si m > m_target: quita aristas al azar (quitar nunca crea triángulos)
    """
    import itertools
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m > m_target:
        # Quitar aristas aleatorias
        edges = list(G.edges())
        rng.shuffle(edges)
        to_remove = max(0, m - m_target)
        for e in edges[:to_remove]:
            G.remove_edge(*e)
        return

    # Necesitamos agregar sin triángulos
    need = m_target - m
    if need <= 0:
        return
    candidates = list(itertools.combinations(range(n), 2))
    rng.shuffle(candidates)
    trials = 0
    added = 0
    for (u, v) in candidates:
        trials += 1
        if trials > max_trials:
            break
        if can_add_without_triangle(G, u, v):
            G.add_edge(u, v)
            added += 1
            if added >= need:
                break

def generate_positive_graph(n: int, p_pos: float, rng: np.random.RandomState) -> nx.Graph:
    """Grafo con triángulos (ER con probabilidad p_pos)."""
    return nx.gnp_random_graph(n, p_pos, seed=int(rng.randint(0, 10**9)))

def generate_negative_hard_graph(n: int, p_base: float, rng: np.random.RandomState,
                                 q_bip: float = None, tol_rel: float = 0.15) -> nx.Graph:
    """
    Genera un negativo "duro":
      1) Parte de bipartito aleatorio (sin triángulos) con prob q_bip (por defecto p_base/2).
      2) Ajusta nº de aristas al objetivo m_target≈p_base * C(n,2) SIN crear triángulos.
    """
    if q_bip is None:
        q_bip = max(0.01, p_base / 2.0)  # densidad inicial en bipartito

    G = random_bipartite_no_triangles(n, q_bip, rng)
    m_target = int(p_base * (n * (n - 1) / 2))
    m_min = int((1 - tol_rel) * m_target)
    m_max = int((1 + tol_rel) * m_target)

    # Ajustar densidad hacia m_target sin crear triángulos
    adjust_density_without_triangles(G, m_target, rng, max_trials=300000)

    # Chequeo de triángulos (debería ser 0)
    tri = triangles_count(G)
    # Permitimos pocos triángulos residuales (idealmente 0)
    if tri != 0:
        # En caso patológico, retiramos aristas de triángulos hasta 0
        # (muy raro con construcción bipartita + agregados seguros).
        while triangles_count(G) > 0:
            # quitar una arista de algún triángulo
            # estrategia simple: retirar arista al azar y continuar
            e = list(G.edges())[int(rng.randint(0, G.number_of_edges()))]
            G.remove_edge(*e)

    # Asegurar densidad dentro de tolerancia
    m = G.number_of_edges()
    if m < m_min:
        adjust_density_without_triangles(G, m_min, rng, max_trials=200000)
    elif m > m_max:
        # quitar aristas al azar
        edges = list(G.edges())
        rng.shuffle(edges)
        to_remove = max(0, m - m_max)
        for e in edges[:to_remove]:
            G.remove_edge(*e)

    return G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--n_por_clase', type=int, default=500)
    ap.add_argument('--p_base', type=float, default=0.12, help='densidad base objetivo')
    ap.add_argument('--p_extra', type=float, default=0.05, help='aumento de densidad para positivos')
    ap.add_argument('--seed', type=int, default=20250925)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    n = args.n
    npos = args.n_por_clase
    nneg = args.n_por_clase

    graphs = []
    labels = []

    # Positivos (con triángulos): p_pos = p_base + p_extra
    p_pos = min(0.99, args.p_base + args.p_extra)

    for _ in range(npos):
        Gp = generate_positive_graph(n, p_pos, rng)
        graphs.append(Gp); labels.append(1)

    # Negativos "duros" (sin triángulos) con densidad ≈ p_base
    for _ in range(nneg):
        Gn = generate_negative_hard_graph(n, args.p_base, rng)
        graphs.append(Gn); labels.append(0)

    # Empaquetar y guardar (lista de listas de aristas)
    edges_list = []
    for G in graphs:
        edges_list.append(np.array(list(G.edges()), dtype=np.int64))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, edges_list=np.array(edges_list, dtype=object), labels=np.array(labels, dtype=np.int64), n=n)
    print(f"[OK] Dataset escrito en: {out} | n={n} | pos={npos}, neg={nneg}")

if __name__ == "__main__":
    main()
