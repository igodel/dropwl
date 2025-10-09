# scripts/exp_c4_compare.py
import argparse
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Core WL / dropWL
from src.core.wl_refinement import compute_wl_colors, color_histograma
from src.core.dropwl_runner import representar_grafo_dropwl_mean

# PyTorch opcional para la rama MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.mlp.mlp_head import MLPHead  # ajusta si tu MLP está en otra ruta
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# ---------------------------------------------------------------------
# Utilidades: dataset
# ---------------------------------------------------------------------
def load_npz_dataset(path: str) -> Tuple[List[nx.Graph], np.ndarray, int]:
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


# ---------------------------------------------------------------------
# Firmas
# ---------------------------------------------------------------------
def wl_signature(G: nx.Graph, t: int, kmax: int) -> np.ndarray:
    """
    Firma WL (histograma normalizado y rellenado hasta kmax) tras t iteraciones.
    Tolerante a compute_wl_colors(t_iter vs t).
    """
    try:
        colores = compute_wl_colors(G, t_iter=t)
    except TypeError:
        try:
            colores = compute_wl_colors(G, t=t)
        except TypeError:
            colores = compute_wl_colors(G, t)  # posicional

    hist = color_histograma(colores)  # dict: color_id -> conteo
    total = sum(hist.values()) if len(hist) > 0 else 1
    vec = [0.0] * kmax
    for cid, cnt in hist.items():
        if 0 <= cid < kmax:
            vec[cid] = cnt / total
    return np.array(vec, dtype=np.float64)


def dropwl_mean_signature(G: nx.Graph, p: float, R: int, t: int, kmax: int, seed_base: int) -> np.ndarray:
    """
    Firma dropWL-mean (vector de dimensión kmax).
    """
    vec = representar_grafo_dropwl_mean(
        G, p=p, R=R, t=t, k_max=kmax, semilla_base=seed_base
    )
    return vec.astype(np.float32)


# ---------------------------------------------------------------------
# Entrenamiento MLP (mean -> MLP)
# ---------------------------------------------------------------------
def _represent_dropwl_mean_batch(graphs, p, R, t, kmax, seed_base=777):
    X = np.stack([
        representar_grafo_dropwl_mean(G, p=p, R=R, t=t, k_max=kmax, semilla_base=seed_base)
        for G in graphs
    ], axis=0)  # [N, kmax]
    return X


def _train_dropwl_mlp_from_mean(Gtr, ytr, Gte, yte,
                                p, R, t, kmax,
                                layers, hidden, d, act,
                                standardize=True,
                                epochs=30, lr=1e-3, patience=8,
                                device="cpu"):
    """
    Entrena dropWL+MLP a partir de la representación 'dropWL-mean' por grafo.
    Arquitectura: MLPHead(kmax -> d) + Linear(d -> 2), CE, Adam, early stopping.

    Retorna:
        acc_train (float), best_acc_test (float)
    """
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch no disponible para dropWL+MLP.")

    # 1) Representaciones
    Xtr = _represent_dropwl_mean_batch(Gtr, p, R, t, kmax)
    Xte = _represent_dropwl_mean_batch(Gte, p, R, t, kmax)

    # 2) Estandarización (recomendada)
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
    acc_tr = 0.0  # última época (para logging)

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


# ---------------------------------------------------------------------
# Ejecución de variantes
# ---------------------------------------------------------------------
def run_variant(graphs, labels, variant_cfg, seed, standardize, device="cpu"):
    """
    Entrena/testea una variante y devuelve dict con métricas + tiempos.
    variant_cfg: dict con keys:
      - name in {"WL","dropWL-mean","dropWL+MLP"}
      - wl_t, wl_kmax
      - dw_p, dw_R, dw_t, dw_kmax
      - mlp_* (si aplica)
    """
    t0 = time.time()

    # split estratificado
    G_train, G_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=0.30, stratify=labels, random_state=seed
    )

    name = variant_cfg["name"]

    # ---------------- WL / dropWL-mean: como antes (firmas fijas + LR)
    if name in ("WL", "dropWL-mean"):
        t_repr0 = time.time()
        X_train, X_test = [], []
        if name == "WL":
            for G in G_train:
                X_train.append(wl_signature(G, t=variant_cfg["wl_t"], kmax=variant_cfg["wl_kmax"]))
            for G in G_test:
                X_test.append(wl_signature(G, t=variant_cfg["wl_t"], kmax=variant_cfg["wl_kmax"]))
        else:  # dropWL-mean
            for G in G_train:
                X_train.append(dropwl_mean_signature(
                    G, p=variant_cfg["dw_p"], R=variant_cfg["dw_R"],
                    t=variant_cfg["dw_t"], kmax=variant_cfg["dw_kmax"],
                    seed_base=seed
                ))
            for G in G_test:
                X_test.append(dropwl_mean_signature(
                    G, p=variant_cfg["dw_p"], R=variant_cfg["dw_R"],
                    t=variant_cfg["dw_t"], kmax=variant_cfg["dw_kmax"],
                    seed_base=seed+1
                ))
        X_train = np.stack(X_train, axis=0)
        X_test  = np.stack(X_test,  axis=0)
        t_repr = time.time() - t_repr0

        # estandarización opcional
        t_fit0 = time.time()
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # Clasificador lineal
        clf = LogisticRegression(
            solver="lbfgs", max_iter=200, random_state=seed, n_jobs=1
        )
        clf.fit(X_train, y_train)
        t_fit = time.time() - t_fit0

        # Evaluación
        t_eval0 = time.time()
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test  = accuracy_score(y_test,  clf.predict(X_test))
        t_eval = time.time() - t_eval0

        return {
            "variant": name,
            "seed": seed,
            "acc_train": float(acc_train),
            "acc_test": float(acc_test),
            "t_repr_s": float(t_repr),
            "t_fit_s": float(t_fit),
            "t_eval_s": float(t_eval),
            "t_total_s": float(time.time() - t0),
            "dim": int(X_train.shape[1]),
        }

    # ---------------- dropWL+MLP: ENTRENAMIENTO real (mean -> MLP)
    elif name == "dropWL+MLP":
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch requerido para dropWL+MLP.")

        t_fit0 = time.time()
        acc_tr, acc_te = _train_dropwl_mlp_from_mean(
            G_train, y_train, G_test, y_test,
            p=variant_cfg["mlp_p"], R=variant_cfg["mlp_R"], t=variant_cfg["mlp_t"], kmax=variant_cfg["mlp_kmax"],
            layers=variant_cfg["mlp_layers"], hidden=variant_cfg["mlp_hidden"],
            d=variant_cfg["mlp_d"], act=variant_cfg["mlp_act"],
            standardize=standardize,
            epochs=30, lr=1e-3, patience=8,
            device=device
        )
        t_fit = time.time() - t_fit0

        return {
            "variant": name,
            "seed": seed,
            "acc_train": float(acc_tr),
            "acc_test": float(acc_te),
            "t_repr_s": 0.0,            # la representación se computa dentro del entrenamiento
            "t_fit_s": float(t_fit),    # tiempo de entrenamiento MLP
            "t_eval_s": 0.0,            # usamos best_acc_test (eval interna)
            "t_total_s": float(time.time() - t0),
            "dim": int(variant_cfg["mlp_d"]),  # dimensión latente del MLP
        }

    else:
        raise ValueError(f"Variante desconocida: {name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Experimento C4: WL vs dropWL-mean vs dropWL+MLP")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250925, 20250926, 20250927])
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")

    # WL
    ap.add_argument("--wl_t", type=int, default=3)
    ap.add_argument("--wl_kmax", type=int, default=20)

    # dropWL-mean
    ap.add_argument("--dw_p", type=float, default=0.1)
    ap.add_argument("--dw_R", type=int, default=50)
    ap.add_argument("--dw_t", type=int, default=3)
    ap.add_argument("--dw_kmax", type=int, default=20)

    # dropWL+MLP
    ap.add_argument("--mlp_p", type=float, default=0.1)
    ap.add_argument("--mlp_R", type=int, default=50)
    ap.add_argument("--mlp_t", type=int, default=3)
    ap.add_argument("--mlp_kmax", type=int, default=20)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu")

    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    graphs, labels, n = load_npz_dataset(args.data)
    print(f"[info] dataset: {len(graphs)} grafos | n={n} | archivo={args.data}")

    # Configs
    wl_cfg = dict(name="WL", wl_t=args.wl_t, wl_kmax=args.wl_kmax)
    dw_cfg = dict(name="dropWL-mean", dw_p=args.dw_p, dw_R=args.dw_R, dw_t=args.dw_t, dw_kmax=args.dw_kmax)
    mlp_cfg = dict(
        name="dropWL+MLP",
        mlp_p=args.mlp_p, mlp_R=args.mlp_R, mlp_t=args.mlp_t, mlp_kmax=args.mlp_kmax,
        mlp_layers=args.mlp_layers, mlp_hidden=args.mlp_hidden, mlp_d=args.mlp_d, mlp_act=args.mlp_act
    )

    variants = [wl_cfg, dw_cfg]
    if HAVE_TORCH:
        variants.append(mlp_cfg)
    else:
        print("[WARN] PyTorch no disponible: se omite dropWL+MLP.")

    rows = []
    for seed in args.seeds:
        for vcfg in variants:
            res = run_variant(
                graphs, labels, vcfg, seed,
                standardize=args.standardize,
                device=args.device
            )
            # guardar hiperparámetros en la fila
            res.update({
                "wl_t": args.wl_t, "wl_kmax": args.wl_kmax,
                "dw_p": args.dw_p, "dw_R": args.dw_R, "dw_t": args.dw_t, "dw_kmax": args.dw_kmax,
                "mlp_p": args.mlp_p, "mlp_R": args.mlp_R, "mlp_t": args.mlp_t, "mlp_kmax": args.mlp_kmax,
                "mlp_layers": args.mlp_layers, "mlp_hidden": args.mlp_hidden, "mlp_d": args.mlp_d, "mlp_act": args.mlp_act
            })
            rows.append(res)
            print(f"[{res['variant']}][seed={seed}] acc_test={res['acc_test']:.4f} "
                  f"(dim={res['dim']}, t_total={res['t_total_s']:.3f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(outdir/"results.csv", index=False)
    print(f"[OK] CSV: {outdir/'results.csv'}")

    # Resumen por variante
    summ = df.groupby("variant")["acc_test"].agg(["mean","std","count"]).reset_index()
    summ.to_csv(outdir/"tables"/"acc_by_variant.csv", index=False)
    print("[info] Resumen por variante:")
    print(summ)

    # Boxplot rápido
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5,4))
        data_box = [df[df["variant"]==v]["acc_test"].values for v in summ["variant"]]
        plt.boxplot(data_box, labels=summ["variant"].tolist())
        plt.ylabel("Accuracy test")
        plt.title("C4: WL vs dropWL variants")
        plt.tight_layout()
        plt.savefig(outdir/"figures"/"box_acc.png", dpi=200)
        print(f"[OK] Figura: {outdir/'figures'/'box_acc.png'}")
    except Exception as e:
        print("[WARN] No se pudo generar figura:", repr(e))


if __name__ == "__main__":
    main()