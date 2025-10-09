#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_paper_c4_suite.py
Orquestador para el experimento C4 (paper). Usa tus modelos:
  - WL (p=0, R=1)
  - dropWL-mean (p>0, R>=1)
  - dropWL+MLP (p>0, R>=1, con MLPHead)

Requisitos previos:
  - scripts/gen_4cycle_controlled.py para generar datasets .npz
  - scripts/exp_c4_compare.py (ya lo tienes) para una corrida simple
Este orquestador llama la lógica equivalente *en local* (no invoca el script),
para componer barridos y consolidar resultados.

Salida:
  results/paper_c4/<dataset_name>/...
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- IMPORTS DE TU REPO (ajustados) ----
from src.core.wl_refinement import compute_wl_colors, color_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean

# PyTorch (opcional para MLP)
try:
    import torch
    from src.mlp.mlp_head import MLPHead  # <- ajustado a tu estructura real
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# ===================== UTILIDADES =====================

def load_npz_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    edges_list = data["edges_list"]
    labels = data["labels"].astype(np.int64)
    n = int(data["n"])
    graphs = []
    for E in edges_list:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (u, v) in E:
            G.add_edge(int(u), int(v))
        graphs.append(G)
    return graphs, labels, n

def wl_signature(G: nx.Graph, t: int, kmax: int) -> np.ndarray:
    colores = compute_wl_colors(G, t_iter=t)
    hist = color_histograma(colores, k_max=kmax, normalizar=True)
    return hist.astype(np.float32)

def dropwl_mean_signature(G: nx.Graph, p: float, R: int, t: int, kmax: int, seed_base: int) -> np.ndarray:
    vec = representar_grafo_dropwl_mean(
        G, p=p, R=R, t=t, k_max=kmax, semilla_base=seed_base
    )
    return vec.astype(np.float32)

def dropwl_mlp_signature(
    G: nx.Graph, p: float, R: int, t: int, kmax: int, seed_base: int,
    mlp_layers: int, mlp_hidden: int, mlp_d: int, mlp_act: str, device: str = "cpu"
) -> np.ndarray:
    """
    Por ejecución: firma WL (kmax) -> MLPHead -> vector d
    Agregación: media sobre R ejecuciones -> vector d final (representación del grafo).
    """
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch/MLP no disponible.")
    act = "relu" if mlp_act == "relu" else "tanh"

    mlp = MLPHead(
        input_dim=kmax, hidden_dim=mlp_hidden, output_dim=mlp_d,
        num_layers=mlp_layers, activation=act
    ).to(device)
    # Inicialización determinista por grafo
    torch.manual_seed(seed_base)
    for p_ in mlp.parameters():
        if p_.ndimension() >= 2:
            torch.nn.init.xavier_uniform_(p_)
        else:
            torch.nn.init.zeros_(p_)
    reps = []
    rng = np.random.default_rng(seed_base)
    # Dropout en aristas (Bernoulli) + WL + MLP
    for r in range(R):
        H = G.copy()
        edges = list(H.edges())
        if p > 0.0:
            mask = rng.random(len(edges)) >= p
            H.remove_edges_from([e for e, keep in zip(edges, mask) if not keep])
        sig = wl_signature(H, t=t, kmax=kmax)  # (kmax,)
        x = torch.from_numpy(sig).float().to(device).view(1, -1)
        with torch.no_grad():
            z = mlp(x)  # (1, d)
        reps.append(z.cpu().numpy().reshape(-1))
    return np.mean(np.stack(reps, axis=0), axis=0).astype(np.float32)

def run_variant(graphs, labels, variant, seed, standardize, cfg, device="cpu"):
    """
    Ejecuta una variante en un dataset:
      variant in {"WL","dropWL-mean","dropWL+MLP"}
    cfg: diccionario con hiperparámetros (ver más abajo).
    """
    t0 = time.time()
    Gtr, Gte, ytr, yte = train_test_split(
        graphs, labels, test_size=0.30, stratify=labels, random_state=seed
    )

    # Representación
    t_repr0 = time.time()
    Xtr, Xte = [], []
    if variant == "WL":
        for G in Gtr:
            Xtr.append(wl_signature(G, t=cfg["t"], kmax=cfg["kmax"]))
        for G in Gte:
            Xte.append(wl_signature(G, t=cfg["t"], kmax=cfg["kmax"]))
    elif variant == "dropWL-mean":
        for G in Gtr:
            Xtr.append(dropwl_mean_signature(G, p=cfg["p"], R=cfg["R"], t=cfg["t"], kmax=cfg["kmax"], seed_base=seed))
        for G in Gte:
            Xte.append(dropwl_mean_signature(G, p=cfg["p"], R=cfg["R"], t=cfg["t"], kmax=cfg["kmax"], seed_base=seed+1))
    elif variant == "dropWL+MLP":
        for G in Gtr:
            Xtr.append(dropwl_mlp_signature(
                G, p=cfg["p"], R=cfg["R"], t=cfg["t"], kmax=cfg["kmax"], seed_base=seed,
                mlp_layers=cfg["layers"], mlp_hidden=cfg["hidden"], mlp_d=cfg["d"], mlp_act=cfg["act"], device=device
            ))
        for G in Gte:
            Xte.append(dropwl_mlp_signature(
                G, p=cfg["p"], R=cfg["R"], t=cfg["t"], kmax=cfg["kmax"], seed_base=seed+1,
                mlp_layers=cfg["layers"], mlp_hidden=cfg["hidden"], mlp_d=cfg["d"], mlp_act=cfg["act"], device=device
            ))
    else:
        raise ValueError("Variante desconocida.")

    Xtr = np.stack(Xtr, axis=0)
    Xte = np.stack(Xte, axis=0)
    t_repr = time.time() - t_repr0

    # Estandarización (opcional, global por columna)
    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    # Clasificador lineal
    t_fit0 = time.time()
    clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=seed, n_jobs=1)
    clf.fit(Xtr, ytr)
    t_fit = time.time() - t_fit0

    # Evaluación
    t_eval0 = time.time()
    acc_tr = accuracy_score(ytr, clf.predict(Xtr))
    acc_te = accuracy_score(yte, clf.predict(Xte))
    t_eval = time.time() - t_eval0

    return {
        "variant": variant,
        "seed": seed,
        "acc_train": float(acc_tr),
        "acc_test": float(acc_te),
        "t_repr_s": float(t_repr),
        "t_fit_s": float(t_fit),
        "t_eval_s": float(t_eval),
        "t_total_s": float(time.time() - t0),
        "dim": int(Xtr.shape[1])
    }

# ===================== MAIN =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="results/paper_c4")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250925, 20250926, 20250927])

    # Conjunto de datasets C4 (ajusta a tus .npz generados desde el paper)
    ap.add_argument("--datasets", type=str, nargs="+", required=True,
                    help="Rutas .npz de C4 (generadas con gen_4cycle_controlled.py)")

    # WL / drop params
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax_eq_n", action="store_true", help="usar kmax = n del dataset")
    ap.add_argument("--kmax", type=int, default=20)

    # Grids de barrido (p,R) y MLP
    ap.add_argument("--grid_p", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--grid_R", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--grid_layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--grid_hidden", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--grid_d", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--grid_act", type=str, nargs="+", default=["relu", "tanh"])
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ALL_ROWS = []

    for ds_path in args.datasets:
        ds = Path(ds_path)
        graphs, labels, n = load_npz_dataset(ds)
        kmax = n if args.kmax_eq_n else args.kmax

        ds_out = out_root / ds.stem
        (ds_out / "tables").mkdir(parents=True, exist_ok=True)
        (ds_out / "figures").mkdir(parents=True, exist_ok=True)

        print(f"[info] dataset={ds.name} | N={len(graphs)} | n={n} | kmax={kmax}")

        # ---- WL (una única config) ----
        for seed in args.seeds:
            cfg_wl = dict(t=args.t, kmax=kmax)
            res = run_variant(graphs, labels, "WL", seed, args.standardize, cfg_wl, device=args.device)
            res.update(dict(dataset=ds.name, p=None, R=1, layers=None, hidden=None, d=None, act=None))
            ALL_ROWS.append(res)
            print(f"  [WL][seed={seed}] acc_test={res['acc_test']:.4f}")

        # ---- dropWL-mean (grid p,R) ----
        for p in args.grid_p:
            for R in args.grid_R:
                for seed in args.seeds:
                    cfg_dw = dict(t=args.t, kmax=kmax, p=p, R=R)
                    res = run_variant(graphs, labels, "dropWL-mean", seed, args.standardize, cfg_dw, device=args.device)
                    res.update(dict(dataset=ds.name, p=p, R=R, layers=None, hidden=None, d=None, act=None))
                    ALL_ROWS.append(res)
                print(f"  [dropWL-mean][p={p},R={R}] OK")

        # ---- dropWL+MLP (grid p,R, layers, hidden, d, act) ----
        if HAVE_TORCH:
            for p in args.grid_p:
                for R in args.grid_R:
                    for layers in args.grid_layers:
                        for hidden in args.grid_hidden:
                            for d in args.grid_d:
                                for act in args.grid_act:
                                    for seed in args.seeds:
                                        cfg_mlp = dict(t=args.t, kmax=kmax, p=p, R=R,
                                                       layers=layers, hidden=hidden, d=d, act=act)
                                        res = run_variant(graphs, labels, "dropWL+MLP", seed, args.standardize, cfg_mlp, device=args.device)
                                        res.update(dict(dataset=ds.name, p=p, R=R, layers=layers, hidden=hidden, d=d, act=act))
                                        ALL_ROWS.append(res)
                                    print(f"  [dropWL+MLP][p={p},R={R},L={layers},H={hidden},d={d},act={act}] OK")
        else:
            print("[WARN] PyTorch no disponible: se omite dropWL+MLP en este orquestador.")

        # ---- Guardar parcial por dataset ----
        DF = pd.DataFrame(ALL_ROWS)
        DF[DF["dataset"] == ds.name].to_csv(ds_out / "results.csv", index=False)

        # Resumen rápido
        sub = DF[DF["dataset"] == ds.name]
        sub.groupby("variant")["acc_test"].agg(["mean","std","count"]).to_csv(ds_out/"tables"/"acc_by_variant.csv")

    # ---- Guardado maestro ----
    DF = pd.DataFrame(ALL_ROWS)
    DF.to_csv(out_root / "results_master.csv", index=False)
    print("[OK] Escritos resultados maestro en:", out_root / "results_master.csv")

if __name__ == "__main__":
    main()
