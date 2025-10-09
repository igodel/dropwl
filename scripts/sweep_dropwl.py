"""
Script: sweep_dropwl.py
Propósito:
    Barrido reproducible sobre el dataset "difícil" (contiene C_k vs NO C_k pero con otros ciclos).
    Compara WL baseline vs dropWL+MLP para distintas combinaciones de hiperparámetros y arquitecturas.
    Registra métricas y tiempos en un CSV.

Uso típico:
    PYTHONPATH=. python scripts/sweep_dropwl.py \
      --data data/cycles_hard_n20_k6.npz \
      --grid_p 0.05 0.1 \
      --grid_R 50 100 \
      --grid_t 3 4 \
      --grid_kmax 20 \
      --grid_layers 1 2 \
      --grid_hidden 64 128 \
      --grid_d 32 64 \
      --epochs 20 \
      --lr 1e-3 \
      --seeds 20250925 20250926 \
      --out results/sweep_hard.csv \
      --include_wl_baseline

Notas:
    - Limita hilos BLAS si tu CPU se satura: export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1
    - El coste crece principalmente con: |train| * R * epochs * n
"""

import sys, time, csv, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import networkx as nx
from typing import List, Tuple

# Modelos / utilidades del proyecto
from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.mlp.mlp_head import MLPHead
from src.train.train_dropwl_mlp import entrenar_epoch, evaluar


# ------------------ Utilidades de carga / representación ------------------

def edges_to_graph(n: int, edges: List[Tuple[int,int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(int(u), int(v)) for (u, v) in edges])
    return g

def representar_wl_baseline(g: nx.Graph, t: int, k_max: int) -> np.ndarray:
    colores = compute_wl_colors(g, numero_de_iteraciones=t, atributo_color_inicial=None)
    firma = construir_firma_histograma(colores, k_max)
    return firma

def load_dataset_npz(path_npz: str):
    data = np.load(path_npz, allow_pickle=True)
    edges_list = list(data["edges_list"])
    labels = np.array(data["labels"], dtype=int)
    n = int(data["n"].item())
    grafos = [edges_to_graph(n, e) for e in edges_list]
    return grafos, labels, n


# ------------------ Barrido ------------------

def run_wl_baseline(g_tr, ytr, g_te, yte, t: int, kmax: int):
    t0 = time.time()

    Xtr = []
    for g in g_tr:
        Xtr.append(representar_wl_baseline(g, t=t, k_max=kmax))
    Xtr = np.vstack(Xtr)

    Xte = []
    for g in g_te:
        Xte.append(representar_wl_baseline(g, t=t, k_max=kmax))
    Xte = np.vstack(Xte)

    t_repr = time.time() - t0

    t1 = time.time()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    t_train = time.time() - t1

    return acc, t_repr, t_train


def run_dropwl_mlp(g_tr, ytr, g_te, yte,
                   p: float, R: int, t_iter: int, kmax: int,
                   mlp_layers: int, hidden: int, d: int,
                   epochs: int, lr: float, seed_exec: int):

    device = torch.device("cpu")
    torch.manual_seed(seed_exec)
    np.random.seed(seed_exec)

    mlp = MLPHead(input_dim=kmax, hidden_dim=hidden, output_dim=d,
                  num_layers=mlp_layers, activation="relu").to(device)
    head = nn.Linear(d, 2).to(device)
    opt = torch.optim.Adam(list(mlp.parameters()) + list(head.parameters()),
                           lr=lr, weight_decay=1e-4)

    # Entrenamiento (re-muestreo por época)
    t0 = time.time()
    for _ in range(epochs):
        _loss = entrenar_epoch(
            grafos=g_tr, labels=ytr,
            p=p, R=R, t=t_iter, k_max=kmax,
            semilla_base=seed_exec, mlp=mlp, head=head, opt=opt, device=device
        )
    t_train = time.time() - t0

    # Evaluación (usa el mismo pipeline de dropWL)
    t1 = time.time()
    acc_tr = evaluar(
        grafos=g_tr, labels=ytr,
        p=p, R=R, t=t_iter, k_max=kmax,
        semilla_base=seed_exec, mlp=mlp, head=head, device=device
    )
    acc_te = evaluar(
        grafos=g_te, labels=yte,
        p=p, R=R, t=t_iter, k_max=kmax,
        semilla_base=seed_exec, mlp=mlp, head=head, device=device
    )
    t_eval = time.time() - t1

    return acc_tr, acc_te, t_train, t_eval


def main():
    ap = argparse.ArgumentParser(description="Barrido de hiperparámetros para dropWL+MLP vs WL baseline (dataset difícil)")
    ap.add_argument("--data", type=str, required=True, help=".npz generado por gen_synthetic_cycles_hard.py")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--grid_p", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--grid_R", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--grid_t", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--grid_kmax", type=int, nargs="+", default=[20])
    ap.add_argument("--grid_layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--grid_hidden", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--grid_d", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250925])
    ap.add_argument("--out", type=str, default="results/sweep.csv")
    ap.add_argument("--include_wl_baseline", action="store_true")
    args = ap.parse_args()

    # (Opcional) limitar hilos para evitar sobreuso de CPU
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # torch.set_num_threads(1); torch.set_num_interop_threads(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Cargar dataset
    grafos, labels, n = load_dataset_npz(args.data)
    print(f"[info] dataset: {len(labels)} grafos | n={n} | archivo={args.data}")

    # CSV
    header = [
        "model", "seed", "p", "R", "t", "kmax",
        "layers", "hidden", "d",
        "epochs", "lr",
        "acc_train", "acc_test",
        "time_repr", "time_train", "time_eval", "time_total"
    ]
    fcsv = open(args.out, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(header)

    # Split (fijamos un split por seed para comparabilidad)
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        idx_tr, idx_te, ytr, yte = train_test_split(
            np.arange(len(grafos)), labels, test_size=args.test_size, random_state=seed, stratify=labels
        )
        g_tr = [grafos[i] for i in idx_tr]
        g_te = [grafos[i] for i in idx_te]

        # -------- WL baseline (opcional) --------
        if args.include_wl_baseline:
            # Convención: p=0, R=1, layers=0, hidden=0, d=0
            for t_iter in args.grid_t:
                for kmax in args.grid_kmax:
                    t0 = time.time()
                    acc_wl, time_repr, time_train = run_wl_baseline(g_tr, ytr, g_te, yte, t=t_iter, kmax=kmax)
                    total = time.time() - t0
                    row = ["WL", seed, 0.0, 1, t_iter, kmax, 0, 0, 0, 0, 0.0,
                           None, acc_wl, time_repr, time_train, 0.0, total]
                    writer.writerow(row); fcsv.flush()
                    print(f"[WL] seed={seed} t={t_iter} kmax={kmax} | acc_test={acc_wl:.4f} | time_total={total:.2f}s")

        # -------- dropWL + MLP --------
        for p in args.grid_p:
            for R in args.grid_R:
                for t_iter in args.grid_t:
                    for kmax in args.grid_kmax:
                        for layers in args.grid_layers:
                            for hidden in args.grid_hidden:
                                for d in args.grid_d:
                                    t0 = time.time()
                                    acc_tr, acc_te, time_train, time_eval = run_dropwl_mlp(
                                        g_tr, ytr, g_te, yte,
                                        p=p, R=R, t_iter=t_iter, kmax=kmax,
                                        mlp_layers=layers, hidden=hidden, d=d,
                                        epochs=args.epochs, lr=args.lr, seed_exec=seed
                                    )
                                    total = time.time() - t0
                                    row = ["dropWL+MLP", seed, p, R, t_iter, kmax,
                                           layers, hidden, d,
                                           args.epochs, args.lr,
                                           acc_tr, acc_te,
                                           None, time_train, time_eval, total]
                                    writer.writerow(row); fcsv.flush()
                                    print(f"[DW+MLP] seed={seed} p={p} R={R} t={t_iter} k={kmax} L={layers} H={hidden} d={d} | "
                                          f"acc_tr={acc_tr:.3f} acc_te={acc_te:.3f} | total={total:.2f}s")

    fcsv.close()
    print("== Barrido finalizado ==")
    print("CSV guardado en:", args.out)


if __name__ == "__main__":
    main()
