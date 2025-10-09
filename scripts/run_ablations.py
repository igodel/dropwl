#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 7 – Ablaciones formales para dropWL+MLP

Experimentos:
  A) Placebo: p=0  (dropWL-mean y MLP deberían colapsar a WL)
  B) Orden:  mean→MLP (nuestro) vs MLP→mean (alternativo)
  C) Estandarización: ON vs OFF

Soporta un dataset .npz (como los que vienes usando) con:
  - edges_list: array de objetos, cada elemento es array de aristas shape (?,2)
  - labels: array [N]
  - n: número de nodos

Salidas:
  results/ablations_phase7/<dataset_name>/results.csv
  results/ablations_phase7/<dataset_name>/figures/*.png (opcionales por corrida)
"""

import argparse
from pathlib import Path
import time
import numpy as np
import networkx as nx

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# torch (solo para MLP)
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# MLP del repo
try:
    from src.mlp.mlp_head import MLPHead
    HAVE_MLPHEAD = True
except Exception:
    HAVE_MLPHEAD = False

# WL del repo
try:
    from src.core.wl_refinement import compute_wl_colors, color_histograma
    HAVE_WL = True
except Exception:
    HAVE_WL = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_npz_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    edges_list = data["edges_list"]
    labels = np.array(data["labels"], dtype=np.int64)
    n = int(data["n"])
    graphs = []
    for arr in edges_list:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (u, v) in arr:
            G.add_edge(int(u), int(v))
        graphs.append(G)
    return graphs, labels, n

def wl_signature_histogram(G: nx.Graph, t: int, k_max: int) -> np.ndarray:
    colores = compute_wl_colors(G, t)
    hist = color_histograma(colores)
    counts = np.array(sorted(hist.values(), reverse=True), dtype=np.float32)
    s = counts.sum()
    if s > 0.0:
        counts = counts / s
    if len(counts) < k_max:
        counts = np.pad(counts, (0, k_max - len(counts)), constant_values=0.0)
    else:
        counts = counts[:k_max]
    return counts

def node_dropout(G: nx.Graph, p: float, rng: np.random.RandomState) -> nx.Graph:
    n = G.number_of_nodes()
    for _ in range(10):
        keep = [v for v in G.nodes() if rng.rand() > p]
        if len(keep) >= 2:
            return G.subgraph(keep).copy()
    keep = rng.choice(list(G.nodes()), size=2, replace=False)
    return G.subgraph(keep).copy()

def represent_dropwl_matrix(graphs, p: float, R: int, t: int, kmax: int, seed_exec: int) -> np.ndarray:
    """
    Devuelve una matriz X_exec de shape (N, R, kmax) con las firmas WL de cada ejecución.
    Para p=0, es determinista (R réplicas idénticas), colapsando a WL.
    """
    rng = np.random.RandomState(seed_exec)
    N = len(graphs)
    X_exec = np.zeros((N, R, kmax), dtype=np.float32)
    for i, G in enumerate(graphs):
        for r in range(R):
            Gi = G if p <= 0.0 else node_dropout(G, p, rng)
            X_exec[i, r, :] = wl_signature_histogram(Gi, t=t, k_max=kmax)
    return X_exec

def standardize_inplace(X: np.ndarray):
    """
    Estandariza features por columna (media 0, var 1) in-place.
    Soporta X de shape (N, D) o (N, R, D) (estandariza sobre N).
    """
    if X.ndim == 2:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        X -= mu
        X /= sd
    elif X.ndim == 3:
        # colapsa R para estimar mu/sd sobre N (promedio por ejecución)
        N, R, D = X.shape
        X2 = X.reshape(N*R, D)
        mu = X2.mean(axis=0, keepdims=True)
        sd = X2.std(axis=0, keepdims=True) + 1e-8
        X -= mu
        X /= sd

def run_wl_baseline(graphs, labels, t, kmax, seed, standardize):
    t0 = time.time()
    X = np.stack([wl_signature_histogram(G, t=t, k_max=kmax) for G in graphs], axis=0)
    if standardize:
        standardize_inplace(X)
    t_repr = time.time() - t0

    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, random_state=seed, stratify=labels)
    clf = LogisticRegression(max_iter=200, solver='lbfgs')
    t1 = time.time()
    clf.fit(Xtr, ytr)
    t_train = time.time() - t1
    acc = accuracy_score(yte, clf.predict(Xte))

    return dict(variant="WL", p=0.0, R=1, order="WL", standardize=bool(standardize),
                acc_test=float(acc), time_repr=float(t_repr), time_train=float(t_train),
                time_total=float(t_repr+t_train))

def run_dropwl_mean(graphs, labels, p, R, t, kmax, seed, seed_exec, standardize):
    t0 = time.time()
    X_exec = represent_dropwl_matrix(graphs, p, R, t, kmax, seed_exec)  # (N,R,D)
    X = X_exec.mean(axis=1)  # mean over executions
    if standardize:
        standardize_inplace(X)
    t_repr = time.time() - t0

    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, random_state=seed, stratify=labels)
    clf = LogisticRegression(max_iter=200, solver='lbfgs')
    t1 = time.time()
    clf.fit(Xtr, ytr)
    t_train = time.time() - t1
    acc = accuracy_score(yte, clf.predict(Xte))

    return dict(variant="dropWL-mean", p=float(p), R=int(R), order="mean", standardize=bool(standardize),
                acc_test=float(acc), time_repr=float(t_repr), time_train=float(t_train),
                time_total=float(t_repr+t_train))

def run_dropwl_mlp(graphs, labels, p, R, t, kmax, seed, seed_exec, standardize,
                   layers, hidden, d, act, epochs, lr, patience, order_variant):
    if not HAVE_TORCH or not HAVE_MLPHEAD:
        raise RuntimeError("PyTorch/MLPHead no disponibles.")

    device = torch.device('cpu')
    # 1) Representaciones por ejecución
    t0 = time.time()
    X_exec = represent_dropwl_matrix(graphs, p, R, t, kmax, seed_exec)  # (N,R,D)
    # Estandarización: depende del orden (para ser justos, aplicamos sobre el objeto que entra al MLP)
    # 2) Split
    Xtr, Xte, ytr, yte = train_test_split(X_exec, labels, test_size=0.3, random_state=seed, stratify=labels)

    # 3) Construir tensores según orden
    act_name = act.lower()
    activation = 'relu' if act_name == 'relu' else 'tanh'
    mlp = MLPHead(input_dim=kmax, hidden_dim=hidden, output_dim=d,
                  num_layers=layers, activation=activation).to(device)
    clf = nn.Linear(d, 2).to(device)

    opt = torch.optim.Adam(list(mlp.parameters()) + list(clf.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    if order_variant == "mean_then_mlp":
        # mean→MLP: primero promediamos R, luego MLP
        Xtr_mean = Xtr.mean(axis=1)  # (N,D)
        Xte_mean = Xte.mean(axis=1)
        if standardize:
            standardize_inplace(Xtr_mean); standardize_inplace(Xte_mean)
        Xtr_t = torch.tensor(Xtr_mean, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte_mean, dtype=torch.float32, device=device)

        ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
        yte_t = torch.tensor(yte, dtype=torch.long, device=device)

        # loop
        best_acc, wait, best_state = -1.0, 0, None
        for ep in range(1, epochs+1):
            mlp.train(); clf.train(); opt.zero_grad()
            Z = mlp(Xtr_t)           # (N,d)
            logits = clf(Z)          # (N,2)
            loss = ce(logits, ytr_t)
            loss.backward(); opt.step()

            mlp.eval(); clf.eval()
            with torch.no_grad():
                Zte = mlp(Xte_t); logits_te = clf(Zte)
                acc = accuracy_score(yte, logits_te.argmax(dim=1).cpu().numpy())
            print(f"[{order_variant} ep={ep:03d}] loss={loss.item():.4f} acc_te={acc:.4f}")
            if acc > best_acc:
                best_acc, best_state, wait = acc, (mlp.state_dict(), clf.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    print("[early-stop]"); break

        if best_state is not None:
            mlp.load_state_dict(best_state[0]); clf.load_state_dict(best_state[1])

        with torch.no_grad():
            Zte = mlp(Xte_t); logits_te = clf(Zte)
            acc = accuracy_score(yte, logits_te.argmax(dim=1).cpu().numpy())

        t_repr = time.time() - t0
        return dict(variant="dropWL+MLP", p=float(p), R=int(R), order="mean->MLP",
                    standardize=bool(standardize), acc_test=float(acc),
                    time_repr=float(t_repr), time_train=0.0, time_total=float(t_repr))

    else:
        # MLP→mean: aplicamos MLP por ejecución y luego promediamos
        if standardize:
            standardize_inplace(Xtr); standardize_inplace(Xte)
        Btr, Rtr, D = Xtr.shape
        Bte, Rte, _ = Xte.shape
        Xtr_t = torch.tensor(Xtr.reshape(Btr*Rtr, D), dtype=torch.float32, device=device)  # (N*R,D)
        Xte_t = torch.tensor(Xte.reshape(Bte*Rte, D), dtype=torch.float32, device=device)
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
        yte_t = torch.tensor(yte, dtype=torch.long, device=device)

        best_acc, wait, best_state = -1.0, 0, None
        for ep in range(1, epochs+1):
            mlp.train(); clf.train(); opt.zero_grad()
            Z = mlp(Xtr_t).view(Btr, Rtr, -1).mean(dim=1)  # (N,d)
            logits = clf(Z); loss = ce(logits, ytr_t)
            loss.backward(); opt.step()

            mlp.eval(); clf.eval()
            with torch.no_grad():
                Zte = mlp(Xte_t).view(Bte, Rte, -1).mean(dim=1)
                logits_te = clf(Zte)
                acc = accuracy_score(yte, logits_te.argmax(dim=1).cpu().numpy())
            print(f"[MLP->mean ep={ep:03d}] loss={loss.item():.4f} acc_te={acc:.4f}")
            if acc > best_acc:
                best_acc, best_state, wait = acc, (mlp.state_dict(), clf.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    print("[early-stop]"); break

        if best_state is not None:
            mlp.load_state_dict(best_state[0]); clf.load_state_dict(best_state[1])

        with torch.no_grad():
            Zte = mlp(Xte_t).view(Bte, Rte, -1).mean(dim=1)
            logits_te = clf(Zte)
            acc = accuracy_score(yte, logits_te.argmax(dim=1).cpu().numpy())

        t_repr = time.time() - t0
        return dict(variant="dropWL+MLP", p=float(p), R=int(R), order="MLP->mean",
                    standardize=bool(standardize), acc_test=float(acc),
                    time_repr=float(t_repr), time_train=0.0, time_total=float(t_repr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    # Hiperparámetros:
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=None, help="Por defecto = n del dataset")
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--standardize", action="store_true")
    # MLP:
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu", choices=["relu","tanh"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early_stop_patience", type=int, default=6)
    # Salida:
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    data_path = Path(args.data)
    graphs, labels, n = load_npz_dataset(data_path)
    kmax = args.kmax if args.kmax is not None else n

    out_base = args.outdir or f"results/ablations_phase7/{data_path.stem}"
    outdir = Path(out_base)
    ensure_dir(outdir)

    print("== Config ==")
    print({**vars(args), "resolved_outdir": str(outdir), "n": n, "kmax": kmax})

    rows = []
    for seed in args.seeds:
        # WL (con y sin standardize)
        for std_flag in [False, True]:
            if not HAVE_WL:
                raise RuntimeError("No pudo importarse src.core.wl_refinement.")
            res_wl = run_wl_baseline(graphs, labels, t=args.t, kmax=kmax, seed=seed, standardize=std_flag)
            res_wl.update(dict(seed=seed, exp="WL_baseline", note=f"std={std_flag}"))
            rows.append(res_wl)

        # A) Placebo p=0: dropWL-mean con p=0
        for std_flag in [False, True]:
            res_dw0 = run_dropwl_mean(graphs, labels, p=0.0, R=args.R, t=args.t, kmax=kmax,
                                      seed=seed, seed_exec=777, standardize=std_flag)
            res_dw0.update(dict(seed=seed, exp="placebo_p0_mean", note=f"std={std_flag}"))
            rows.append(res_dw0)

        # A) Placebo p=0: dropWL+MLP con p=0 (si TORCH)
        if HAVE_TORCH and HAVE_MLPHEAD:
            for std_flag in [False, True]:
                for order in ["mean_then_mlp", "mlp_then_mean"]:
                    res_mlp0 = run_dropwl_mlp(graphs, labels, p=0.0, R=args.R, t=args.t, kmax=kmax,
                                              seed=seed, seed_exec=777, standardize=std_flag,
                                              layers=args.mlp_layers, hidden=args.mlp_hidden, d=args.mlp_d,
                                              act=args.mlp_act, epochs=args.epochs, lr=args.lr,
                                              patience=args.early_stop_patience, order_variant=order)
                    res_mlp0.update(dict(seed=seed, exp="placebo_p0_mlp", note=f"std={std_flag} {order}"))
                    rows.append(res_mlp0)
        else:
            print("[WARN] PyTorch/MLP no disponible: se omite placebo p=0 para MLP.")

        # B) Orden (p real): mean->MLP vs MLP->mean
        if HAVE_TORCH and HAVE_MLPHEAD:
            for std_flag in [False, True]:
                for order in ["mean_then_mlp", "mlp_then_mean"]:
                    res_mlp = run_dropwl_mlp(graphs, labels, p=args.p, R=args.R, t=args.t, kmax=kmax,
                                             seed=seed, seed_exec=777, standardize=std_flag,
                                             layers=args.mlp_layers, hidden=args.mlp_hidden, d=args.mlp_d,
                                             act=args.mlp_act, epochs=args.epochs, lr=args.lr,
                                             patience=args.early_stop_patience, order_variant=order)
                    res_mlp.update(dict(seed=seed, exp="order_p>0", note=f"std={std_flag} {order}"))
                    rows.append(res_mlp)
        else:
            print("[WARN] PyTorch/MLP no disponible: se omite experimento de orden.")

        # C) Estandarización ON/OFF para dropWL-mean (p>0)
        for std_flag in [False, True]:
            res_dwp = run_dropwl_mean(graphs, labels, p=args.p, R=args.R, t=args.t, kmax=kmax,
                                      seed=seed, seed_exec=777, standardize=std_flag)
            res_dwp.update(dict(seed=seed, exp="std_effect_mean", note=f"std={std_flag}"))
            rows.append(res_dwp)

    import pandas as pd
    DF = pd.DataFrame(rows)
    DF.to_csv(outdir / "results.csv", index=False)
    print("[OK] Escritos resultados en:", outdir / "results.csv")


if __name__ == "__main__":
    main()
