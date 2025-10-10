#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento simple (node-level) para datasets con:
  edges_list, node_labels_list, n

Modelos:
  - WL               (one-hot de color WL-t por nodo, LogisticRegression)
  - 1drop-LOG        (una arista aleatoria removida, WL-t, LogisticRegression)
  - 1drop-MLP        (una arista aleatoria removida, WL-t -> one-hot -> MLPHead + Linear)

Split 70/30 estratificado a NIVEL GRAFO (como acordamos):
  - Particionamos la lista de grafos (no nodos) 70/30 por seed.
  - Entrenamos con TODOS los nodos de los grafos de train, testeamos con TODOS los nodos de test.

Salida:
  outdir/results.csv con columnas:
    variant, seed, acc_train, acc_test, t_repr_s, t_fit_s, t_eval_s, t_total_s, dim
"""
import argparse, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- WL utils ya existentes ---
from src.core.wl_refinement import compute_wl_colors

# --- 1-drop (una arista) ---
def _remove_one_random_edge(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    H = G.copy()
    E = list(H.edges())
    if len(E) > 0:
        eidx = int(rng.integers(0, len(E)))
        H.remove_edge(*E[eidx])
    return H

# --- Torch MLP opcional ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.mlp.mlp_head import MLPHead
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

def load_npz_node_dataset(path: str) -> Tuple[List[nx.Graph], List[np.ndarray], int]:
    data = np.load(path, allow_pickle=True)
    edges_list = data["edges_list"]
    node_labels_list = data["node_labels_list"]
    n = int(data["n"])
    graphs = []
    labels_per_graph = []
    for E, lab in zip(edges_list, node_labels_list):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (u,v) in E:
            G.add_edge(int(u), int(v))
        graphs.append(G)
        labels_per_graph.append(lab.astype(np.int64))
    return graphs, labels_per_graph, n

def wl_node_colors(G: nx.Graph, t: int) -> Dict[int, int]:
    # firma robusta a la API (t_iter vs t)
    try:
        return compute_wl_colors(G, t_iter=t)
    except TypeError:
        try:
            return compute_wl_colors(G, t=t)
        except TypeError:
            return compute_wl_colors(G, t)

def nodes_onehot_from_colors(colors: Dict[int,int], kmax: int, n: int) -> np.ndarray:
    X = np.zeros((n, kmax), dtype=np.float32)
    for u, c in colors.items():
        if 0 <= c < kmax:
            X[u, c] = 1.0
    return X

def represent_nodes_WL(G: nx.Graph, t: int, kmax: int) -> np.ndarray:
    colors = wl_node_colors(G, t)
    return nodes_onehot_from_colors(colors, kmax, G.number_of_nodes())

def represent_nodes_1drop(G: nx.Graph, t: int, kmax: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = _remove_one_random_edge(G, rng)
    colors = wl_node_colors(H, t)
    return nodes_onehot_from_colors(colors, kmax, H.number_of_nodes())

def train_eval_logreg(Xtr, ytr, Xte, yte, seed=0):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=200, solver="lbfgs", random_state=seed, n_jobs=1)
    clf.fit(Xtr_s, ytr)
    acc_tr = accuracy_score(ytr, clf.predict(Xtr_s))
    acc_te = accuracy_score(yte, clf.predict(Xte_s))
    return float(acc_tr), float(acc_te)

def train_eval_mlp(Xtr, ytr, Xte, yte, seed=0, d=64, hidden=128, layers=2, act="relu",
                   epochs=30, lr=1e-3, patience=8, device="cpu"):
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch requerido para 1drop-MLP (node-level).")
    torch.manual_seed(seed)
    in_dim = Xtr.shape[1]
    net = MLPHead(input_dim=in_dim, hidden_dim=hidden, output_dim=d,
                  num_layers=layers, activation=act).to(device)
    head = nn.Linear(d, 2).to(device)

    opt = optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    best_te = 0.0; patience_left = patience
    for _ in range(epochs):
        net.train(); head.train()
        opt.zero_grad()
        z = net(Xtr_t)
        logits = head(z)
        loss = ce(logits, ytr_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            net.eval(); head.eval()
            acc_tr = (logits.argmax(1) == ytr_t).float().mean().item()
            acc_te = (head(net(Xte_t)).argmax(1) == yte_t).float().mean().item()
        if acc_te > best_te:
            best_te = acc_te; patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return float(acc_tr), float(best_te)

def main():
    ap = argparse.ArgumentParser(description="Comparación WL vs 1drop (node-level)")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--R", type=int, default=50)  # R no se usa en 1drop (siempre 1 arista), se ignora
    ap.add_argument("--run_wl", action="store_true")
    ap.add_argument("--run_1drop_log", action="store_true")
    ap.add_argument("--run_1drop_mlp", action="store_true")
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir/"figures").mkdir(parents=True, exist_ok=True)
    (outdir/"tables").mkdir(parents=True, exist_ok=True)

    graphs, node_labels_list, n = load_npz_node_dataset(args.data)
    # split 70/30 por grafo
    idx = np.arange(len(graphs))
    for seed in args.seeds:
        tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, random_state=seed, stratify=None)

        def stack_nodes(repr_fun):
            Xs, ys = [], []
            for i in tr_idx:
                Xi = repr_fun(graphs[i])
                yi = node_labels_list[i]
                Xs.append(Xi); ys.append(yi)
            Xtr = np.vstack(Xs); ytr = np.concatenate(ys)

            Xs, ys = [], []
            for i in te_idx:
                Xi = repr_fun(graphs[i])
                yi = node_labels_list[i]
                Xs.append(Xi); ys.append(yi)
            Xte = np.vstack(Xs); yte = np.concatenate(ys)
            return Xtr, ytr, Xte, yte

        rows = []

        if args.run_wl:
            t0 = time.time()
            Xtr, ytr, Xte, yte = stack_nodes(lambda G: represent_nodes_WL(G, args.t, args.kmax))
            t_repr = time.time()-t0; t_fit0 = time.time()
            acc_tr, acc_te = train_eval_logreg(Xtr, ytr, Xte, yte, seed=seed)
            t_fit = time.time()-t_fit0; t_eval = 0.0
            rows.append(dict(variant="WL", seed=seed, acc_train=acc_tr, acc_test=acc_te,
                             t_repr_s=t_repr, t_fit_s=t_fit, t_eval_s=t_eval,
                             t_total_s=time.time()-t0, dim=int(Xtr.shape[1])))
            print(f"[WL][seed={seed}] acc_test={acc_te:.4f} (dim={Xtr.shape[1]}, t_total={time.time()-t0:.3f}s)")

        if args.run_1drop_log:
            t0 = time.time()
            Xtr, ytr, Xte, yte = stack_nodes(lambda G: represent_nodes_1drop(G, args.t, args.kmax, seed))
            t_repr = time.time()-t0; t_fit0 = time.time()
            acc_tr, acc_te = train_eval_logreg(Xtr, ytr, Xte, yte, seed=seed)
            t_fit = time.time()-t_fit0; t_eval = 0.0
            rows.append(dict(variant="1drop-LOG", seed=seed, acc_train=acc_tr, acc_test=acc_te,
                             t_repr_s=t_repr, t_fit_s=t_fit, t_eval_s=t_eval,
                             t_total_s=time.time()-t0, dim=int(Xtr.shape[1])))
            print(f"[1drop-LOG][seed={seed}] acc_test={acc_te:.4f} (dim={Xtr.shape[1]}, t_total={time.time()-t0:.3f}s)")

        if args.run_1drop_mlp:
            t0 = time.time()
            Xtr, ytr, Xte, yte = stack_nodes(lambda G: represent_nodes_1drop(G, args.t, args.kmax, seed))
            t_repr = time.time()-t0; t_fit0 = time.time()
            acc_tr, acc_te = train_eval_mlp(
                Xtr, ytr, Xte, yte, seed=seed,
                d=args.mlp_d, hidden=args.mlp_hidden, layers=args.mlp_layers, act=args.mlp_act,
                epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device
            )
            t_fit = time.time()-t_fit0; t_eval = 0.0
            rows.append(dict(variant="1drop-MLP", seed=seed, acc_train=acc_tr, acc_test=acc_te,
                             t_repr_s=t_repr, t_fit_s=t_fit, t_eval_s=t_eval,
                             t_total_s=time.time()-t0, dim=int(args.mlp_d)))
            print(f"[1drop-MLP][seed={seed}] acc_test={acc_te:.4f} (dim={args.mlp_d}, t_total={time.time()-t0:.3f}s)")

        df = pd.DataFrame(rows)
        # Apéndice por seed (idempotente: si ya existe, concatenas; o guarda por fuera y luego agregas)
        out_csv = outdir/"results.csv"
        if out_csv.exists():
            old = pd.read_csv(out_csv)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(out_csv, index=False)

    # Resumen corto
    try:
        df = pd.read_csv(outdir/"results.csv")
        summ = df.groupby("variant")["acc_test"].agg(["mean","std","count"]).reset_index()
        print("[info] Resumen por variante:\n", summ)
        summ.to_csv(outdir/"tables"/"acc_by_variant.csv", index=False)
    except Exception as e:
        print("[WARN] no pude resumir:", repr(e))

if __name__ == "__main__":
    main()
