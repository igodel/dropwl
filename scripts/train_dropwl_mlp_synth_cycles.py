"""
Script: train_dropwl_mlp_synth_cycles.py
Propósito:
    Entrenar y evaluar WL baseline vs dropWL+MLP en el dataset sintético "contiene C_k".

Uso:
    # Entrenamiento típico
    python scripts/train_dropwl_mlp_synth_cycles.py \
        --data data/cycles_n20_k6.npz \
        --t 3 --kmax 20 \
        --p 0.05 --R 100 \
        --hidden 128 --d 64 --num_layers 2 --epochs 20 --lr 1e-3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List, Tuple
import time
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean
from src.mlp.mlp_head import MLPHead
from src.train.train_dropwl_mlp import entrenar_epoch, evaluar


def edges_to_graph(n: int, edges: List[Tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(int(u), int(v)) for (u, v) in edges])
    return g


def representar_wl_baseline(g: nx.Graph, t: int, k_max: int) -> np.ndarray:
    colores = compute_wl_colors(g, numero_de_iteraciones=t, atributo_color_inicial=None)
    firma = construir_firma_histograma(colores, k_max)
    return firma


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WL baseline vs dropWL+MLP en dataset sintético C_k")
    parser.add_argument("--data", type=str, required=True, help="ruta al .npz generado por gen_synthetic_cycles.py")
    parser.add_argument("--t", type=int, default=3, help="iteraciones de 1-WL")
    parser.add_argument("--kmax", type=int, default=20, help="dimensión de la firma por ejecución")
    parser.add_argument("--p", type=float, default=0.05, help="prob. de dropout por nodo (dropWL)")
    parser.add_argument("--R", type=int, default=100, help="número de ejecuciones por grafo (dropWL)")
    parser.add_argument("--hidden", type=int, default=128, help="tamaño de capa oculta MLP")
    parser.add_argument("--d", type=int, default=64, help="dimensión de salida MLP")
    parser.add_argument("--num_layers", type=int, default=2, help="número de capas MLP (1 o 2)")
    parser.add_argument("--activation", type=str, default="relu", help="relu o tanh")
    parser.add_argument("--epochs", type=int, default=20, help="épocas de entrenamiento dropWL+MLP")
    parser.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje (Adam)")
    parser.add_argument("--seed", type=int, default=20250925, help="semilla reproducible")
    args = parser.parse_args()

    print("== Configuración ==")
    print(vars(args))

    # Cargar dataset
    data = np.load(args.data, allow_pickle=True)
    edges_list = list(data["edges_list"])
    labels = np.array(data["labels"], dtype=int)
    n = int(data["n"].item())
    print(f"[info] dataset: {len(labels)} grafos | n={n} | archivo={args.data}")

    # Reconstruir grafos
    grafos: List[nx.Graph] = []
    i = 0
    while i < len(edges_list):
        g = edges_to_graph(n, edges_list[i])
        grafos.append(g)
        i = i + 1

    # Split
    idx_tr, idx_te, ytr, yte = train_test_split(
        np.arange(len(grafos)), labels, test_size=0.3, random_state=args.seed, stratify=labels
    )
    g_tr = [grafos[i] for i in idx_tr]
    g_te = [grafos[i] for i in idx_te]
    print(f"[info] split: train={len(ytr)}, test={len(yte)}")

    # ===== WL baseline =====
    tA0 = time.time()
    Xtr_wl = []
    i = 0
    while i < len(g_tr):
        Xtr_wl.append(representar_wl_baseline(g_tr[i], t=args.t, k_max=args.kmax))
        i = i + 1
    Xtr_wl = np.vstack(Xtr_wl)

    Xte_wl = []
    j = 0
    while j < len(g_te):
        Xte_wl.append(representar_wl_baseline(g_te[j], t=args.t, k_max=args.kmax))
        j = j + 1
    Xte_wl = np.vstack(Xte_wl)
    tA1 = time.time()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr_wl, ytr)
    ypred_wl = clf.predict(Xte_wl)
    acc_wl = accuracy_score(yte, ypred_wl)
    tA2 = time.time()

    print("\n== WL baseline ==")
    print(f"Representación WL: {tA1 - tA0:.3f}s | Entrenamiento LR: {tA2 - tA1:.3f}s")
    print(f"Accuracy test (WL): {acc_wl:.4f}")

    # ===== dropWL + MLP =====
    device = torch.device("cpu")
    mlp = MLPHead(input_dim=args.kmax, hidden_dim=args.hidden, output_dim=args.d,
                  num_layers=args.num_layers, activation=args.activation).to(device)
    head = nn.Linear(args.d, 2).to(device)
    opt = torch.optim.Adam(list(mlp.parameters()) + list(head.parameters()),
                           lr=args.lr, weight_decay=1e-4)

    tB0 = time.time()
    ep = 1
    while ep <= args.epochs:
        _loss = entrenar_epoch(
            grafos=g_tr, labels=ytr,
            p=args.p, R=args.R, t=args.t, k_max=args.kmax,
            semilla_base=args.seed, mlp=mlp, head=head, opt=opt, device=device
        )
        ep = ep + 1
    tB1 = time.time()

    acc_dw_tr = evaluar(
        grafos=g_tr, labels=ytr,
        p=args.p, R=args.R, t=args.t, k_max=args.kmax,
        semilla_base=args.seed, mlp=mlp, head=head, device=device
    )
    acc_dw_te = evaluar(
        grafos=g_te, labels=yte,
        p=args.p, R=args.R, t=args.t, k_max=args.kmax,
        semilla_base=args.seed, mlp=mlp, head=head, device=device
    )
    tB2 = time.time()

    print("\n== dropWL + MLP ==")
    print(f"Entrenamiento (épocas={args.epochs}, R={args.R}, p={args.p}): {tB1 - tB0:.3f}s")
    print(f"Evaluación: {tB2 - tB1:.3f}s")
    print(f"Accuracy train (dropWL+MLP): {acc_dw_tr:.4f}")
    print(f"Accuracy test  (dropWL+MLP): {acc_dw_te:.4f}")

    # ===== Resumen =====
    print("\n== Resumen paralelo ==")
    print(f"WL baseline   -> test acc: {acc_wl:.4f}  (p=0, R=1, sin MLP)")
    print(f"dropWL + MLP  -> test acc: {acc_dw_te:.4f}  (p={args.p}, R={args.R}, MLP activo)")
    print(f"Tiempo total: {time.time() - tA0:.3f}s")


if __name__ == "__main__":
    main()
