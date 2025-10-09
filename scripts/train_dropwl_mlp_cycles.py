"""
Script: train_dropwl_mlp_cycles
Propósito:
    Entrenar y evaluar dropWL + MLP en el problema sintético C8 vs 2C4.

Uso:
    python scripts/train_dropwl_mlp_cycles.py --epochs 20 --p 0.1 --R 100
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import List, Tuple
import argparse
import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from sklearn.model_selection import train_test_split

from src.mlp.mlp_head import MLPHead
from src.train.train_dropwl_mlp import entrenar_epoch, evaluar


def construir_c8() -> nx.Graph:
    return nx.cycle_graph(8)


def construir_2c4() -> nx.Graph:
    G1 = nx.cycle_graph(4)
    G2 = nx.cycle_graph(4)
    G = nx.disjoint_union(G1, G2)
    return G


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento dropWL + MLP en C8 vs 2C4")
    parser.add_argument("--n_por_clase", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--t", type=int, default=3)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20250925)
    parser.add_argument("--seed_exec", type=int, default=777)
    args = parser.parse_args()

    print("== Configuración ==")
    print(vars(args))

    # Semillas
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset y split
    grafos, y = dataset_c8_vs_2c4(n_por_clase=args.n_por_clase, semilla=args.seed)
    g_tr, g_te, y_tr, y_te = train_test_split(grafos, y, test_size=0.3, random_state=42, stratify=y)
    print(f"[info] train={len(y_tr)}, test={len(y_te)}")

    # Modelo
    device = torch.device("cpu")
    mlp = MLPHead(input_dim=args.kmax, hidden_dim=args.hidden, output_dim=args.d,
                  num_layers=args.num_layers, activation=args.activation).to(device)
    head = nn.Linear(args.d, 2).to(device)

    # Optimizador
    opt = torch.optim.Adam(list(mlp.parameters()) + list(head.parameters()),
                           lr=args.lr, weight_decay=1e-4)

    # Entrenamiento
    epoca = 1
    while epoca <= args.epochs:
        loss = entrenar_epoch(
            grafos=g_tr, labels=y_tr,
            p=args.p, R=args.R, t=args.t, k_max=args.kmax,
            semilla_base=args.seed_exec,
            mlp=mlp, head=head, opt=opt, device=device
        )
        acc_tr = evaluar(
            grafos=g_tr, labels=y_tr,
            p=args.p, R=args.R, t=args.t, k_max=args.kmax,
            semilla_base=args.seed_exec,
            mlp=mlp, head=head, device=device
        )
        acc_te = evaluar(
            grafos=g_te, labels=y_te,
            p=args.p, R=args.R, t=args.t, k_max=args.kmax,
            semilla_base=args.seed_exec,
            mlp=mlp, head=head, device=device
        )
        print(f"[época {epoca:03d}] loss={loss:.4f} | acc_train={acc_tr:.3f} | acc_test={acc_te:.3f}")
        epoca = epoca + 1

    print("== Final ==")
    print("Accuracy train:", round(float(acc_tr), 4))
    print("Accuracy test :", round(float(acc_te), 4))


if __name__ == "__main__":
    main()
