"""
Script: compare_wl_vs_dropwl.py
Propósito:
    Comparar WL baseline (p=0, R=1, sin MLP) vs dropWL+MLP (p>0, R grande, con MLP)
    en el problema sintético C8 vs 2C4. Imprime resultados en paralelo.

Uso típico:
    python scripts/compare_wl_vs_dropwl.py \
        --n_por_clase 100 --t 3 --kmax 8 \
        --p 0.1 --R 100 \
        --hidden 128 --d 64 --num_layers 2 --epochs 20 --lr 1e-3

Notas:
    - WL baseline usa LogisticRegression de scikit-learn (sin MLP).
    - dropWL+MLP usa el pipeline PyTorch de la Fase 5 (re-muestreo por época).
"""

# --- Ajuste de rutas para importar "src/..." ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Imports estándar ---
import time
from typing import List, Tuple
import argparse
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Módulos del proyecto ---
from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean

import torch
import torch.nn as nn
from src.mlp.mlp_head import MLPHead
from src.train.train_dropwl_mlp import entrenar_epoch, evaluar


# --------------- Generación del dataset C8 vs 2C4 ---------------

def construir_c8() -> nx.Graph:
    g = nx.cycle_graph(8)
    return g

def construir_2c4() -> nx.Graph:
    g1 = nx.cycle_graph(4)
    g2 = nx.cycle_graph(4)
    g = nx.disjoint_union(g1, g2)
    return g

def dataset_c8_vs_2c4(n_por_clase: int, semilla: int) -> Tuple[List[nx.Graph], np.ndarray]:
    rng = np.random.RandomState(semilla)
    grafos: List[nx.Graph] = []
    etiquetas = []

    i = 0
    while i < n_por_clase:
        grafos.append(construir_c8())
        etiquetas.append(0)
        i = i + 1

    j = 0
    while j < n_por_clase:
        grafos.append(construir_2c4())
        etiquetas.append(1)
        j = j + 1

    idx = np.arange(len(grafos))
    rng.shuffle(idx)
    grafos = [grafos[k] for k in idx]
    y = np.array([etiquetas[k] for k in idx], dtype=int)
    return grafos, y


# --------------- Representación WL baseline (sin dropout) ---------------

def representar_wl_baseline(g: nx.Graph, t: int, k_max: int) -> np.ndarray:
    """
    p=0, R=1, sin MLP: sólo 1-WL sobre el grafo original y firma histograma.
    """
    colores = compute_wl_colors(g, numero_de_iteraciones=t, atributo_color_inicial=None)
    firma = construir_firma_histograma(colores, k_max)
    return firma


# --------------- Main: comparación lado a lado ---------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Comparación WL baseline vs dropWL+MLP en C8 vs 2C4")
    # Dataset y WL
    parser.add_argument("--n_por_clase", type=int, default=100, help="número de grafos por clase")
    parser.add_argument("--t", type=int, default=3, help="iteraciones 1-WL")
    parser.add_argument("--kmax", type=int, default=8, help="dimensión de la firma por ejecución")
    parser.add_argument("--seed", type=int, default=20250925, help="semilla dataset/split")

    # dropWL
    parser.add_argument("--p", type=float, default=0.1, help="probabilidad de dropout por nodo")
    parser.add_argument("--R", type=int, default=100, help="número de ejecuciones por grafo")
    parser.add_argument("--seed_exec", type=int, default=777, help="semilla base para ejecuciones")

    # MLP + entrenamiento
    parser.add_argument("--hidden", type=int, default=128, help="tamaño de capa oculta del MLP")
    parser.add_argument("--d", type=int, default=64, help="dimensión de salida del MLP")
    parser.add_argument("--num_layers", type=int, default=2, help="número de capas del MLP (1 o 2)")
    parser.add_argument("--activation", type=str, default="relu", help="relu o tanh")
    parser.add_argument("--epochs", type=int, default=20, help="épocas de entrenamiento (dropWL+MLP)")
    parser.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje (dropWL+MLP)")

    args = parser.parse_args()

    print("== Configuración ==")
    print(vars(args))

    # Semillas reproducibles
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Sugerencia de entorno (opcional): limitar hilos BLAS si notas lentitud
    # import os
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # torch.set_num_threads(1); torch.set_num_interop_threads(1)

    # ---------------- Dataset ----------------
    t0 = time.time()
    grafos, y = dataset_c8_vs_2c4(n_por_clase=args.n_por_clase, semilla=args.seed)
    Xtr_idx, Xte_idx, ytr, yte = train_test_split(
        np.arange(len(grafos)), y, test_size=0.3, random_state=42, stratify=y
    )
    g_tr = [grafos[i] for i in Xtr_idx]
    g_te = [grafos[i] for i in Xte_idx]
    print(f"[t+{time.time()-t0:.3f}s] Dataset listo: total={len(grafos)} | train={len(ytr)} | test={len(yte)}")

    # ========== BLOQUE A: WL baseline (p=0, R=1, sin MLP) ==========
    tA0 = time.time()
    X_wl_tr = []
    i = 0
    while i < len(g_tr):
        X_wl_tr.append(representar_wl_baseline(g_tr[i], t=args.t, k_max=args.kmax))
        i = i + 1
    X_wl_tr = np.vstack(X_wl_tr)

    X_wl_te = []
    j = 0
    while j < len(g_te):
        X_wl_te.append(representar_wl_baseline(g_te[j], t=args.t, k_max=args.kmax))
        j = j + 1
    X_wl_te = np.vstack(X_wl_te)
    tA1 = time.time()

    clf_wl = LogisticRegression(max_iter=1000)
    clf_wl.fit(X_wl_tr, ytr)
    ypred_wl = clf_wl.predict(X_wl_te)
    acc_wl = accuracy_score(yte, ypred_wl)
    tA2 = time.time()

    print("\n== WL baseline ==")
    print(f"Representación WL: {tA1-tA0:.3f}s | Entrenamiento LR: {tA2-tA1:.3f}s")
    print(f"Accuracy test (WL sin dropout): {acc_wl:.4f}")

    # ========== BLOQUE B: dropWL+MLP (p>0, R grande, con MLP) ==========
    # Modelo (MLP + cabeza lineal 2 clases)
    device = torch.device("cpu")
    mlp = MLPHead(
        input_dim=args.kmax,
        hidden_dim=args.hidden,
        output_dim=args.d,
        num_layers=args.num_layers,
        activation=args.activation
    ).to(device)
    head = nn.Linear(args.d, 2).to(device)
    opt = torch.optim.Adam(list(mlp.parameters()) + list(head.parameters()),
                           lr=args.lr, weight_decay=1e-4)

    # Entrenamiento con re-muestreo por época (idéntico a Fase 5)
    from src.train.train_dropwl_mlp import entrenar_epoch, evaluar

    tB0 = time.time()
    ep = 1
    while ep <= args.epochs:
        _loss = entrenar_epoch(
            grafos=g_tr, labels=ytr,
            p=args.p, R=args.R, t=args.t, k_max=args.kmax,
            semilla_base=args.seed_exec,
            mlp=mlp, head=head, opt=opt, device=device
        )
        ep = ep + 1
    tB1 = time.time()

    # Evaluación (mismo pipeline dropWL)
    acc_dw_tr = evaluar(
        grafos=g_tr, labels=ytr,
        p=args.p, R=args.R, t=args.t, k_max=args.kmax,
        semilla_base=args.seed_exec,
        mlp=mlp, head=head, device=device
    )
    acc_dw_te = evaluar(
        grafos=g_te, labels=yte,
        p=args.p, R=args.R, t=args.t, k_max=args.kmax,
        semilla_base=args.seed_exec,
        mlp=mlp, head=head, device=device
    )
    tB2 = time.time()

    print("\n== dropWL + MLP ==")
    print(f"Entrenamiento (épocas={args.epochs}, R={args.R}, p={args.p}): {tB1-tB0:.3f}s")
    print(f"Evaluación: {tB2-tB1:.3f}s")
    print(f"Accuracy train (dropWL+MLP): {acc_dw_tr:.4f}")
    print(f"Accuracy test  (dropWL+MLP): {acc_dw_te:.4f}")

    # ----------------- Resumen paralelo -----------------
    print("\n== Resumen paralelo (WL vs dropWL+MLP) ==")
    print(f"WL baseline   -> test acc: {acc_wl:.4f}  (p=0, R=1, sin MLP)")
    print(f"dropWL + MLP  -> test acc: {acc_dw_te:.4f}  (p={args.p}, R={args.R}, MLP activo)")
    print(f"Tiempo total: {time.time()-t0:.3f}s")


if __name__ == "__main__":
    main()
