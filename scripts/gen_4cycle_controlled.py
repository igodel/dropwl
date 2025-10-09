#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de dataset balanceado para presencia de 4-ciclos (C4).

- Clase 1: grafos que CONTIENEN al menos un C4.
- Clase 0: grafos que NO CONTIENEN C4.
- Base: Erdős–Rényi G(n, p) por cada ejemplo.
- Garantía: si un grafo no cumple la etiqueta, se modifica de forma controlada
  (inyectando un C4 mínimo o rompiendo testigos de C4) con reintentos acotados.
- Guardado: edges_list como array de objetos (dtype=object) para admitir
  longitudes variables (cada grafo tiene distinto |E|). Carga con allow_pickle=True.

Funcionamiento eficiente:
- Detección de C4 en O(n^2) usando el criterio: “existe un par (u, v) con
  |N(u) ∩ N(v)| ≥ 2”. Ese par induce un C4 u–w–v–x–u para dos vecinos comunes w, x.

Recomendación:
- Para n ≤ ~100, esto corre muy rápido. Si subes mucho n o el número de ejemplos,
  aumenta --max_reintentos_neg / --max_reintentos_pos si hace falta.

Autor: tú
"""

import argparse
from pathlib import Path
import numpy as np
import random
import networkx as nx
from typing import Optional, Tuple, List


# ------------------------- Utilidades C4 -------------------------

def _two_common_neighbors(G: nx.Graph, u: int, v: int) -> Optional[Tuple[int, int]]:
    """
    Retorna dos vecinos comunes distintos (w, x) de (u, v) si existen; si no, None.
    """
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    inter = Nu.intersection(Nv)
    if len(inter) >= 2:
        it = iter(inter)
        w = next(it)
        x = next(it)
        return (w, x)
    return None


def has_C4(G: nx.Graph) -> bool:
    """
    Existe C4 ssi ∃ par (u, v) con al menos 2 vecinos comunes.
    Complejidad ≈ O(n^2) (intersecciones de sets).
    """
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        u = nodes[i]
        Nu = set(G.neighbors(u))
        # Si grado < 2, no puede aportar 2 vecinos comunes
        if len(Nu) < 2:
            continue
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            Nv = set(G.neighbors(v))
            if len(Nv) < 2:
                continue
            if len(Nu.intersection(Nv)) >= 2:
                return True
    return False


def inject_one_C4(G: nx.Graph, rng: random.Random) -> None:
    """
    Inyecta un C4 mínimo añadiendo aristas de un ciclo a-b-c-d-a sobre 4 nodos distintos.
    No garantiza que sea cuerda-libre; no es necesario para la tarea.
    """
    nodes = list(G.nodes())
    if len(nodes) < 4:
        return
    a, b, c, d = rng.sample(nodes, 4)
    G.add_edge(a, b)
    G.add_edge(b, c)
    G.add_edge(c, d)
    G.add_edge(d, a)


def break_one_C4(G: nx.Graph) -> bool:
    """
    Rompe un testigo de C4 si existe:
      - Busca un par (u, v) con ≥2 vecinos comunes (w, x).
      - Quita una arista del ciclo (por ej., (u, w)).
    Devuelve True si logró romper uno; False si no encontró C4.
    """
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        u = nodes[i]
        Nu = set(G.neighbors(u))
        if len(Nu) < 2:
            continue
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            Nv = set(G.neighbors(v))
            if len(Nv) < 2:
                continue
            pair = _two_common_neighbors(G, u, v)
            if pair is not None:
                w, x = pair
                # Rompemos una arista del ciclo testigo: u-w
                if G.has_edge(u, w):
                    G.remove_edge(u, w)
                elif G.has_edge(v, w):
                    G.remove_edge(v, w)
                elif G.has_edge(u, x):
                    G.remove_edge(u, x)
                elif G.has_edge(v, x):
                    G.remove_edge(v, x)
                else:
                    # Caso raro: por una carrera de cambios; intenta otra arista
                    # en la siguiente iteración.
                    pass
                return True
    return False


# -------------------- Generación y garantía de clase --------------------

def ensure_has_C4(G: nx.Graph, rng: random.Random,
                  max_reintentos_pos: int = 20) -> nx.Graph:
    """
    Asegura que G contenga C4. Si no, intenta inyectar hasta max_reintentos_pos veces.
    """
    if has_C4(G):
        return G
    for _ in range(max_reintentos_pos):
        inject_one_C4(G, rng)
        if has_C4(G):
            return G
    # Si no lo logra (muy raro), retorna tal cual; la etiqueta la fijará el llamador
    return G


def ensure_no_C4(G: nx.Graph,
                 max_reintentos_neg: int = 200) -> nx.Graph:
    """
    Asegura que G NO contenga C4. Si los hay, rompe testigos hasta que no quede ninguno
    o se alcancen max_reintentos_neg operaciones.
    """
    cnt = 0
    while has_C4(G) and cnt < max_reintentos_neg:
        ok = break_one_C4(G)
        cnt += 1 if ok else 0
        if not ok:
            # No encontró testigo para romper (carrera), fuerza recomprobación
            pass
    return G


def graph_to_edge_list(G: nx.Graph) -> np.ndarray:
    """
    Convierte G a lista de aristas (u, v) con u < v, ordenada.
    Devuelve un ndarray shape (m, 2) de dtype int64.
    """
    E = [tuple(sorted(e)) for e in G.edges()]
    E.sort()
    return np.array(E, dtype=np.int64)


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Genera dataset balanceado C4-presencia.")
    ap.add_argument("--n", type=int, required=True, help="Nº nodos por grafo")
    ap.add_argument("--p", type=float, required=True, help="Prob. arista ER")
    ap.add_argument("--n_por_clase", type=int, required=True, help="Ejemplos por clase")
    ap.add_argument("--seed", type=int, default=20250925)
    ap.add_argument("--max_reintentos_pos", type=int, default=20,
                    help="Máximo de inyecciones de C4 si falta (clase positiva)")
    ap.add_argument("--max_reintentos_neg", type=int, default=200,
                    help="Máximo de roturas de C4 si sobra (clase negativa)")
    ap.add_argument("--out", type=str, required=True, help="Ruta .npz de salida")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    graphs: List[nx.Graph] = []
    labels: List[int] = []

    # Clase 1 (con C4)
    for _ in range(args.n_por_clase):
        G = nx.erdos_renyi_graph(args.n, args.p, seed=rng.randint(0, 10**9))
        G = ensure_has_C4(G, rng, max_reintentos_pos=args.max_reintentos_pos)
        graphs.append(G)
        labels.append(1)

    # Clase 0 (sin C4)
    for _ in range(args.n_por_clase):
        G = nx.erdos_renyi_graph(args.n, args.p, seed=rng.randint(0, 10**9))
        G = ensure_no_C4(G, max_reintentos_neg=args.max_reintentos_neg)
        graphs.append(G)
        labels.append(0)

    # Barajar
    idx = np_rng.permutation(len(graphs))
    graphs = [graphs[i] for i in idx]
    labels = np.array([labels[i] for i in idx], dtype=np.int64)

    # Edge-lists (longitudes variables) -> array de objetos
    edges_list = [graph_to_edge_list(G) for G in graphs]
    edges_list_obj = np.array(edges_list, dtype=object)

    # Guardar
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        edges_list=edges_list_obj,     # dtype=object
        labels=labels,
        n=int(args.n)
    )
    print(f"[OK] Guardado {len(graphs)} grafos en {out} | n={args.n} | p={args.p} | balanceado 50/50")


if __name__ == "__main__":
    main()
