#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento explícito para C4 (data_paper/*.npz):
- WL baseline (LR)
- dropWL-mean (LR)
- dropWL+MLP (PyTorch) con control de epochs, lr y early stopping.

Uso:
PYTHONPATH=. python scripts/train_c4_mlp_explicit.py \
  --data data_paper/c4_n8_p030_bal.npz \
  --seeds 20250925 20250926 \
  --t 3 --kmax 8 \
  --p 0.1 --R 100 \
  --mlp_layers 2 --mlp_hidden 64 --mlp_d 64 --mlp_act relu \
  --epochs 30 --lr 1e-3 --patience 8 \
  --standardize \
  --outdir results/_c4_mlp_explicit_n8
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Tuple
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.wl_refinement import compute_wl_colors, color_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean
from src.mlp.mlp_head import MLPHead
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.dropwl_runner import representar_grafo_dropwl_mean
from src.mlp.mlp_head import MLPHead

def _represent_dropwl_mean_batch(graphs, p, R, t, kmax, seed_base=777):
    X = np.stack([
        representar_grafo_dropwl_mean(G, p=p, R=R, t=t, k_max=kmax, semilla_base=seed_base)
        for G in graphs
    ], axis=0)  # shape [N, kmax]
    return X

def _train_dropwl_mlp_from_mean(Gtr, ytr, Gte, yte,
                                p, R, t, kmax,
                                layers, hidden, d, act,
                                standardize=True,
                                epochs=30, lr=1e-3, patience=8,
                                device="cpu"):
    # 1) Representaciones (mean ya agrega R internamente)
    Xtr = _represent_dropwl_mean_batch(Gtr, p, R, t, kmax)
    Xte = _represent_dropwl_mean_batch(Gte, p, R, t, kmax)

    # 2) Estandarización opcional (recomendada)
    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    # 3) MLP + clasificador lineal
    net = MLPHead(input_dim=kmax, hidden_dim=hidden, output_dim=d,
                  num_layers=layers, activation=act).to(device)
    clf = nn.Linear(d, 2).to(device)
    opt = optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    best_te = 0.0
    patience_left = patience

    for _ in range(epochs):
        net.train(); clf.train()
        opt.zero_grad()
        z = net(Xtr_t)          # [N, d]
        logits = clf(z)         # [N, 2]
        loss = ce(logits, ytr_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            net.eval(); clf.eval()
            # train
            pred_tr = logits.argmax(dim=1).detach().cpu().numpy()
            acc_tr = accuracy_score(ytr, pred_tr)
            # test
            pred_te = clf(net(Xte_t)).argmax(dim=1).detach().cpu().numpy()
            acc_te = accuracy_score(yte, pred_te)

        if acc_te > best_te:
            best_te = acc_te
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return float(acc_tr), float(best_te)

def wl_signature(G: nx.Graph, t: int, kmax: int) -> np.ndarray:
    # llamada robusta a compute_wl_colors: t_iter/t/posicional
    try:
        colores = compute_wl_colors(G, t_iter=t)
    except TypeError:
        try:
            colores = compute_wl_colors(G, t=t)
        except TypeError:
            colores = compute_wl_colors(G, t)
    hist = color_histograma(colores)
    total = sum(hist.values()) if hist else 1
    vec = np.zeros(kmax, dtype=np.float64)
    for cid, cnt in hist.items():
        if 0 <= cid < kmax:
            vec[cid] = cnt / total
    return vec

def load_npz_dataset(path: str) -> Tuple[List[np.ndarray], np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    edges_list = data["edges_list"]  # dtype=object
    labels = data["labels"].astype(np.int64)
    n = int(data["n"])
    return edges_list, labels, n

def build_graphs(edges_list, n) -> List[nx.Graph]:
    out = []
    for E in edges_list:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(E)
        out.append(G)
    return out

def get_splits(Gs: List[nx.Graph], y: np.ndarray, seed: int) -> Tuple[List[nx.Graph], List[nx.Graph], np.ndarray, np.ndarray]:
    idx = np.arange(len(Gs))
    tr, te = train_test_split(idx, test_size=200, train_size=1000, random_state=seed, stratify=y)
    return [Gs[i] for i in tr], [Gs[i] for i in te], y[tr], y[te]

def represent_wl(Gs: List[nx.Graph], t: int, kmax: int) -> np.ndarray:
    X = np.stack([wl_signature(G, t, kmax) for G in Gs], axis=0)
    return X

def represent_dropwl_mean(Gs: List[nx.Graph], p: float, R: int, t: int, kmax: int, seed_base: int=777) -> np.ndarray:
    X = np.stack([representar_grafo_dropwl_mean(G, p=p, R=R, t=t, k_max=kmax, semilla_base=seed_base) for G in Gs], axis=0)
    return X

def train_lr(Xtr, ytr, Xte, yte) -> Tuple[float, float]:
    clf = LogisticRegression(max_iter=200, solver='lbfgs')
    clf.fit(Xtr, ytr)
    return accuracy_score(ytr, clf.predict(Xtr)), accuracy_score(yte, clf.predict(Xte))

def train_mlp_drop(Gtr: List[nx.Graph], Gte: List[nx.Graph], ytr: np.ndarray, yte: np.ndarray,
                   p: float, R: int, t: int, kmax: int,
                   layers: int, hidden: int, d: int, act: str,
                   epochs: int, lr: float, patience: int,
                   standardize: bool, device: str="cpu") -> Tuple[float, float]:
    # 1) Representación por ejecuciones (R réplicas, firma kmax) -> tensor [N, R, kmax]
    rng_seed = 777
    reps_tr = [representar_grafo_dropwl_mean(G, p=p, R=R, t=t, k_max=kmax, semilla_base=rng_seed) for G in Gtr]
    reps_te = [representar_grafo_dropwl_mean(G, p=p, R=R, t=t, k_max=kmax, semilla_base=rng_seed) for G in Gte]
    Xtr = np.stack(reps_tr, axis=0)  # [Ntr, kmax] (ya es mean en nuestro runner) -> lo elevamos a [Ntr,1,kmax]
    Xte = np.stack(reps_te, axis=0)

    # como nuestro representar_grafo_dropwl_mean ya promedia R, simulamos “R=1” a la entrada del MLP
    Xtr = Xtr[:, None, :]  # [Ntr,1,kmax]
    Xte = Xte[:, None, :]

    # 2) opcional: estandarizar por dimensión kmax (antes del MLP)
    if standardize:
        sc = StandardScaler()
        Xtr_2d = sc.fit_transform(Xtr.reshape(Xtr.shape[0], -1))
        Xte_2d = sc.transform(Xte.reshape(Xte.shape[0], -1))
        Xtr = Xtr_2d.reshape(Xtr.shape[0], 1, -1)
        Xte = Xte_2d.reshape(Xte.shape[0], 1, -1)

    # 3) MLP compartido por ejecución (aquí 1), agregado por media (identidad)
    in_dim = kmax
    net = MLPHead(input_dim=in_dim, hidden_dim=hidden, output_dim=d, num_layers=layers, activation=act).to(device)
    clf = nn.Linear(d, 2).to(device)

    opt = optim.Adam(list(net.parameters())+list(clf.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)  # [N,1,kmax]
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    best_te = 0.0
    patience_left = patience

    for ep in range(1, epochs+1):
        net.train(); clf.train()
        opt.zero_grad()
        z_tr = net(Xtr_t.squeeze(1))       # [N,kmax] -> [N,d]
        logits = clf(z_tr)                  # [N,2]
        loss = ce(logits, ytr_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            net.eval(); clf.eval()
            pred_tr = logits.argmax(dim=1).cpu().numpy()
            acc_tr = (pred_tr == ytr).mean()

            z_te = net(Xte_t.squeeze(1))
            log_te = clf(z_te)
            pred_te = log_te.argmax(dim=1).cpu().numpy()
            acc_te = (pred_te == yte).mean()

        if acc_te > best_te:
            best_te = acc_te
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return float(acc_tr), float(best_te)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    edges_list, labels, n = load_npz_dataset(args.data)
    Gs = build_graphs(edges_list, n)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in args.seeds:
        Gtr, Gte, ytr, yte = get_splits(Gs, labels, seed)

        # WL
        Xtr_wl = represent_wl(Gtr, t=args.t, kmax=args.kmax)
        Xte_wl = represent_wl(Gte, t=args.t, kmax=args.kmax)
        if args.standardize:
            sc = StandardScaler()
            Xtr_wl = sc.fit_transform(Xtr_wl); Xte_wl = sc.transform(Xte_wl)
        acc_tr, acc_te = train_lr(Xtr_wl, ytr, Xte_wl, yte)
        rows.append(dict(variant="WL", seed=seed, acc_train=acc_tr, acc_test=acc_te))

        # dropWL-mean
        Xtr_dw = represent_dropwl_mean(Gtr, p=args.p, R=args.R, t=args.t, kmax=args.kmax)
        Xte_dw = represent_dropwl_mean(Gte, p=args.p, R=args.R, t=args.t, kmax=args.kmax)
        if args.standardize:
            sc = StandardScaler()
            Xtr_dw = sc.fit_transform(Xtr_dw); Xte_dw = sc.transform(Xte_dw)
        acc_tr, acc_te = train_lr(Xtr_dw, ytr, Xte_dw, yte)
        rows.append(dict(variant="dropWL-mean", seed=seed, acc_train=acc_tr, acc_test=acc_te))

        # dropWL+MLP
        acc_tr, acc_te = train_mlp_drop(Gtr, Gte, ytr, yte,
                                        p=args.p, R=args.R, t=args.t, kmax=args.kmax,
                                        layers=args.mlp_layers, hidden=args.mlp_hidden, d=args.mlp_d, act=args.mlp_act,
                                        epochs=args.epochs, lr=args.lr, patience=args.patience,
                                        standardize=args.standardize, device="cpu")
        rows.append(dict(variant="dropWL+MLP", seed=seed, acc_train=acc_tr, acc_test=acc_te))

    df = pd.DataFrame(rows)
    df.to_csv(out/"results.csv", index=False)
    print("== Resumen ==")
    print(df.groupby("variant")["acc_test"].agg(["mean","std","count"]).round(4))

if __name__ == "__main__":
    main()
