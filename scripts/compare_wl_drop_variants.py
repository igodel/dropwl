# compare_wl_drop_variants.py
# Uso (solo WL + dropWL-mean, sin torch; útil en ARM):
#   PYTHONPATH=. python compare_wl_drop_variants.py \
#     --data data/triangles_hard_n20.npz \
#     --seeds 20250925 20250926 \
#     --run_wl --run_drop_mean \
#     --wl_t 3 --wl_kmax 20 \
#     --dw_p 0.1 --dw_R 50 --dw_t 3 --dw_kmax 20 \
#     --standardize --time_each_step \
#     --outdir results/triangles_compare_arm --learning_curves
#
# Uso (WL + dropWL-mean + dropWL+MLP; requiere torch):
#   PYTHONPATH=. python compare_wl_drop_variants.py \
#     --data data/triangles_hard_n20.npz \
#     --seeds 20250925 20250926 \
#     --run_wl --run_drop_mean --run_drop_mlp \
#     --wl_t 3 --wl_kmax 20 \
#     --dw_p 0.1 --dw_R 50 --dw_t 3 --dw_kmax 20 \
#     --mlp_p 0.1 --mlp_R 50 --mlp_t 3 --mlp_kmax 20 \
#     --mlp_layers 1 --mlp_hidden 64 --mlp_d 32 --mlp_act relu \
#     --epochs 30 --lr 1e-3 --early_stop_patience 8 \
#     --standardize --time_each_step \
#     --outdir results/triangles_compare_full --learning_curves
#
# Salidas:
#   <outdir>/results.csv                         (una fila por seed x variante)
#   <outdir>/tables/*.csv                        (resúmenes, pruebas pareadas)
#   <outdir>/figures/*.png + report_compare.pdf  (figuras y PDF consolidado)
#
# Notas:
#  - Implementa 1-WL (refinamiento) y firmas (histograma normalizado, kmax).
#  - dropWL-mean: promedia R firmas bajo node-dropout (p).
#  - dropWL+MLP: cada firma pasa por MLP compartido; se promedian las salidas.
#  - Curvas de aprendizaje: accuracy vs fracción de train (mismo test).
#  - Import de torch es condicional y totalmente interno a la función de entrenamiento.

import argparse, time, json, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============== Utilidades =================

def set_seed(seed: int):
    np.random.seed(seed)

def load_npz_dataset(path: str) -> Tuple[List[nx.Graph], np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    edges_list = data["edges_list"]
    labels = data["labels"].astype(int)
    n = int(data["n"]) if "n" in data.files else None
    graphs = []
    for edges in edges_list:
        G = nx.Graph()
        # si n estaba en metadatos, fuerza nodos 0..n-1 para tamaño consistente
        if n is not None:
            G.add_nodes_from(range(n))
        for (u, v) in edges:
            G.add_edge(int(u), int(v))
        graphs.append(G)
    if n is None:
        # inferir n máximo
        n = max(g.number_of_nodes() for g in graphs)
    return graphs, labels, n

# ------------- 1-WL refinement + firma (histograma) ----------------

def wl_refinement_colors(G: nx.Graph, t: int) -> dict:
    """
    Refinamiento 1-WL (t iteraciones) sobre grafos no dirigidos sin atributos.
    Devuelve mapa: nodo -> color_id (entero) de la última iteración.
    """
    # colores iniciales (todos 0)
    colors = {u: 0 for u in G.nodes()}
    for _ in range(t):
        # para cada nodo, construir firma (color propio, multiconjunto de vecinos)
        signatures = {}
        for u in G.nodes():
            neigh_colors = sorted(colors[v] for v in G.neighbors(u))
            # tupla determinista
            sig = (colors[u], tuple(neigh_colors))
            signatures[u] = sig
        # reindexar a enteros compactos
        uniq = {}
        next_color = 0
        new_colors = {}
        for u, sig in signatures.items():
            if sig not in uniq:
                uniq[sig] = next_color
                next_color += 1
            new_colors[u] = uniq[sig]
        colors = new_colors
    return colors

def signature_histogram(G: nx.Graph, t: int, kmax: int) -> np.ndarray:
    """
    Firma por histograma:
      - Contar frecuencias de colores finales.
      - Ordenar frecuencias de mayor a menor (invariante a permutaciones de IDs).
      - Normalizar (suma 1).
      - Rellenar con ceros hasta kmax.
    """
    colors = wl_refinement_colors(G, t)
    counts = {}
    for c in colors.values():
        counts[c] = counts.get(c, 0) + 1
    freqs = sorted(counts.values(), reverse=True)
    vec = np.array(freqs, dtype=float)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    if len(vec) < kmax:
        vec = np.pad(vec, (0, kmax - len(vec)))
    else:
        vec = vec[:kmax]
    return vec

# ------------- dropWL-mean (sin MLP) ----------------

def sample_induced_subgraph(G: nx.Graph, p_drop: float, rng: np.random.RandomState, min_nodes: int = 1) -> nx.Graph:
    """
    Node-dropout independiente con prob. p_drop.
    Garantiza al menos 'min_nodes' (reintentos acotados).
    """
    nodes = list(G.nodes())
    for _ in range(20):
        keep = [u for u in nodes if rng.rand() > p_drop]
        if len(keep) >= min_nodes:
            return G.subgraph(keep).copy()
    # si todo falla, devuelve el grafo original (caso extremo)
    return G.copy()

def dropwl_mean_features(G: nx.Graph, p: float, R: int, t: int, kmax: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    reps = []
    for r in range(R):
        H = sample_induced_subgraph(G, p, rng, min_nodes=1)
        sig = signature_histogram(H, t, kmax)
        reps.append(sig)
    reps = np.stack(reps, axis=0)  # (R, kmax)
    return reps.mean(axis=0)       # (kmax,)

# ------------- dropWL+MLP (definición interna, sin dependencia global de torch) ----------------

def train_dropwl_mlp(
    graphs: List[nx.Graph], y: np.ndarray, idx_train: np.ndarray, idx_test: np.ndarray,
    p: float, R: int, t: int, kmax: int,
    layers: int, hidden: int, d: int, activation: str,
    epochs: int, lr: float, weight_decay: float = 0.0, early_stop_patience: int = 0, seed: int = 0
):
    """
    Entrenamiento end-to-end (solo si --run_drop_mlp):
      - Para cada grafo: R subgrafos vía node-dropout p; 1-WL -> firma (kmax).
      - Cada firma pasa por un MLP compartido; se promedian las salidas (mean).
      - Capa lineal final y pérdida CrossEntropy.
    Nota: `torch` y la clase MLP se definen aquí adentro para que el módulo sea importable sin PyTorch.
    """
    import torch
    import numpy as np

    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_dim: int, hidden: int, output_dim: int, layers: int = 1, activation: str = "relu"):
            super().__init__()
            act = torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh()
            if layers <= 1:
                self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
            else:
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden),
                    act,
                    torch.nn.Linear(hidden, output_dim)
                )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    num_classes = int(len(np.unique(y)))
    head = SimpleMLP(kmax, hidden, d, layers=layers, activation=activation).to(device)
    clf  = torch.nn.Linear(d, num_classes).to(device)

    params = list(head.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    def forward_graph(G, seed_local):
        rng = np.random.RandomState(seed_local)
        feats = []
        for _ in range(R):
            H = sample_induced_subgraph(G, p, rng, min_nodes=1)
            sig = signature_histogram(H, t, kmax)  # (kmax,)
            feats.append(sig)
        X = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32, device=device)  # (R,kmax)
        Z = head(X)                   # (R,d)
        z = Z.mean(dim=0)             # (d,)
        out = clf(z.view(1, -1))      # (1, C)
        return out.squeeze(0)         # (C,)

    best_val = None
    patience = 0
    t0 = time.time()
    for epoch in range(1, epochs+1):
        head.train(); clf.train()
        for i in idx_train:
            out = forward_graph(graphs[i], seed + epoch + i)
            yi = torch.tensor(int(y[i]), dtype=torch.long, device=device)
            loss = loss_fn(out.view(1,-1), yi.view(1))
            opt.zero_grad(); loss.backward(); opt.step()

        # evaluación rápida para early stopping (en test como proxy simple)
        head.eval(); clf.eval()
        with torch.no_grad():
            pred_test = []
            for i in idx_test:
                out = forward_graph(graphs[i], seed + 54321 + i)
                pred_test.append(int(out.argmax().item()))
            acc_test = (np.array(pred_test) == np.array(y[idx_test])).mean()
        if early_stop_patience > 0:
            crit = acc_test
            if (best_val is None) or (crit > best_val):
                best_val = crit; patience = 0
                best_state = {"head": head.state_dict(), "clf": clf.state_dict()}
            else:
                patience += 1
                if patience >= early_stop_patience:
                    head.load_state_dict(best_state["head"])
                    clf.load_state_dict(best_state["clf"])
                    break

    train_time = time.time() - t0

    # evaluación final separada
    t1 = time.time()
    head.eval(); clf.eval()
    with torch.no_grad():
        pred_train = []
        for i in idx_train:
            out = forward_graph(graphs[i], seed + 999 + i)
            pred_train.append(int(out.argmax().item()))
        acc_train = (np.array(pred_train) == np.array(y[idx_train])).mean()

        pred_test = []
        for i in idx_test:
            out = forward_graph(graphs[i], seed + 777 + i)
            pred_test.append(int(out.argmax().item()))
        acc_test = (np.array(pred_test) == np.array(y[idx_test])).mean()
    eval_time = time.time() - t1

    return {
        "acc_train": float(acc_train),
        "acc_test": float(acc_test),
        "time_repr": 0.0,
        "time_train": float(train_time),
        "time_eval": float(eval_time),
        "time_total": float(train_time + eval_time),
    }

# ------------- Curvas de aprendizaje ----------------

def learning_curve_variant(variant_name, graphs, labels, idx_train, idx_test, variant_cfg, standardize=False, seed=0):
    """
    Entrena la variante con fracciones crecientes de entrenamiento y mide accuracy en test.
    Retorna dict con arrays 'fracs' y 'acc_test'.
    """
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    accs = []

    # extraer X_train_full / X_test_full sólo para WL y drop-mean; para MLP entrenar directo
    if variant_name in ["WL", "dropWL-mean"]:
        # precomputar representaciones
        if variant_name == "WL":
            t = variant_cfg["t"]; kmax = variant_cfg["kmax"]
            X = np.stack([signature_histogram(G, t, kmax) for G in graphs], axis=0)
        else:
            p = variant_cfg["p"]; R = variant_cfg["R"]; t = variant_cfg["t"]; kmax = variant_cfg["kmax"]
            seed_base = 4242 + seed
            X = np.stack([dropwl_mean_features(G, p, R, t, kmax, seed_base+i) for i, G in enumerate(graphs)], axis=0)
        y = labels
        X_train_full, X_test = X[idx_train], X[idx_test]
        y_train_full, y_test = y[idx_train], y[idx_test]

        scaler = StandardScaler() if standardize else None
        if scaler is not None:
            X_train_full = scaler.fit_transform(X_train_full)
            X_test = scaler.transform(X_test)

        n_train = len(X_train_full)
        for f in fracs:
            m = max(1, int(math.ceil(f * n_train)))
            X_train = X_train_full[:m]
            y_train = y_train_full[:m]

            clf = LogisticRegression(max_iter=variant_cfg.get("max_iter", 200), solver=variant_cfg.get("solver", "lbfgs"))
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accs.append(accuracy_score(y_test, y_pred))

    else:  # dropWL+MLP (entrenar con subsets crecientes)
        p = variant_cfg["p"]; R = variant_cfg["R"]; t = variant_cfg["t"]; kmax = variant_cfg["kmax"]
        layers = variant_cfg["layers"]; hidden = variant_cfg["hidden"]; d = variant_cfg["d"]; act = variant_cfg["act"]
        epochs = variant_cfg["epochs"]; lr = variant_cfg["lr"]; wd = variant_cfg.get("wd", 0.0); patience = variant_cfg.get("patience", 0)
        for f in fracs:
            m = max(1, int(math.ceil(f * len(idx_train))))
            sub_idx_train = idx_train[:m]
            res = train_dropwl_mlp(graphs, labels, sub_idx_train, idx_test, p, R, t, kmax, layers, hidden, d, act, epochs, lr, wd, patience, seed)
            accs.append(res["acc_test"])

    return {"fracs": np.array(fracs), "acc_test": np.array(accs)}

# ------------- Pruebas apareadas ----------------

def bootstrap_ci_mean_delta(deltas: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> Tuple[float, float, float]:
    rng = np.random.RandomState(123)
    n = len(deltas)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boots.append(np.mean(deltas[idx]))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return (float(np.mean(deltas)), float(lo), float(hi))

def paired_permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 10000) -> float:
    rng = np.random.RandomState(1234)
    d = x - y
    obs = abs(np.mean(d))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(d))
        stat = abs(np.mean(d * signs))
        if stat >= obs:
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return float(pval)

# =============== Pipeline principal =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--time_each_step", action="store_true")
    ap.add_argument("--outdir", type=str, required=True)

    # WL baseline
    ap.add_argument("--run_wl", action="store_true")
    ap.add_argument("--wl_t", type=int, default=3)
    ap.add_argument("--wl_kmax", type=int, default=20)
    ap.add_argument("--wl_solver", type=str, default="lbfgs")
    ap.add_argument("--wl_max_iter", type=int, default=200)

    # dropWL-mean
    ap.add_argument("--run_drop_mean", action="store_true")
    ap.add_argument("--dw_p", type=float, default=0.1)
    ap.add_argument("--dw_R", type=int, default=50)
    ap.add_argument("--dw_t", type=int, default=3)
    ap.add_argument("--dw_kmax", type=int, default=20)
    ap.add_argument("--dw_reduce", type=str, default="mean")  # (implementado: mean)
    ap.add_argument("--dw_solver", type=str, default="lbfgs")
    ap.add_argument("--dw_max_iter", type=int, default=200)

    # dropWL+MLP
    ap.add_argument("--run_drop_mlp", action="store_true")
    ap.add_argument("--mlp_p", type=float, default=0.1)
    ap.add_argument("--mlp_R", type=int, default=50)
    ap.add_argument("--mlp_t", type=int, default=3)
    ap.add_argument("--mlp_kmax", type=int, default=20)
    ap.add_argument("--mlp_layers", type=int, default=1)
    ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--mlp_d", type=int, default=32)
    ap.add_argument("--mlp_act", type=str, default="relu", choices=["relu","tanh"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--early_stop_patience", type=int, default=0)

    # Learning curves
    ap.add_argument("--learning_curves", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir/"figures").mkdir(parents=True, exist_ok=True)
    (outdir/"tables").mkdir(parents=True, exist_ok=True)

    graphs, labels, n = load_npz_dataset(args.data)
    _ = int(len(np.unique(labels)))  # num_classes, no usado fuera

    # Chequeo opcional de torch si se solicita MLP (sin afectar ARM cuando no se usa)
    if args.run_drop_mlp:
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Se solicitó --run_drop_mlp pero no se pudo importar torch. "
                "Instala PyTorch o ejecuta sin esta variante (--run_wl/--run_drop_mean)."
            ) from e

    # Acumuladores de resultados por seed/variante
    rows = []

    # Para curvas de aprendizaje: dict acumulado por variante
    lc_data = {
        "WL": [],
        "dropWL-mean": [],
        "dropWL+MLP": []
    }

    # Bucle por semillas
    for seed in args.seeds:
        set_seed(seed)

        # split estratificado fijo por seed
        idx = np.arange(len(graphs))
        idx_train, idx_test = train_test_split(idx, test_size=args.test_size, stratify=labels, random_state=seed)

        # ===== Variante 1: WL baseline =====
        if args.run_wl:
            t0 = time.time()
            X = np.stack([signature_histogram(graphs[i], args.wl_t, args.wl_kmax) for i in idx], axis=0)
            t_repr = time.time() - t0 if args.time_each_step else 0.0

            X_train = X[idx_train]; y_train = labels[idx_train]
            X_test  = X[idx_test];  y_test  = labels[idx_test]

            scaler = StandardScaler() if args.standardize else None
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

            t1 = time.time()
            clf = LogisticRegression(max_iter=args.wl_max_iter, solver=args.wl_solver)
            clf.fit(X_train, y_train)
            t_train = time.time() - t1 if args.time_each_step else 0.0

            t2 = time.time()
            y_pred_train = clf.predict(X_train)
            y_pred_test  = clf.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test  = accuracy_score(y_test,  y_pred_test)
            t_eval = time.time() - t2 if args.time_each_step else 0.0

            rows.append({
                "seed": seed, "variant": "WL",
                "acc_train": acc_train, "acc_test": acc_test,
                "time_repr": t_repr, "time_train": t_train, "time_eval": t_eval,
                "time_total": t_repr + t_train + t_eval,
                "t": args.wl_t, "kmax": args.wl_kmax
            })

            # Learning curve
            if args.learning_curves:
                lc = learning_curve_variant(
                    "WL", graphs, labels, idx_train, idx_test,
                    {"t": args.wl_t, "kmax": args.wl_kmax, "max_iter": args.wl_max_iter, "solver": args.wl_solver},
                    standardize=args.standardize, seed=seed
                )
                lc_data["WL"].append(lc)

        # ===== Variante 2: dropWL-mean =====
        if args.run_drop_mean:
            t0 = time.time()
            # Representaciones promedio bajo dropout
            X = np.stack([dropwl_mean_features(graphs[i], args.dw_p, args.dw_R, args.dw_t, args.dw_kmax, seed + i)
                          for i in idx], axis=0)
            t_repr = time.time() - t0 if args.time_each_step else 0.0

            X_train = X[idx_train]; y_train = labels[idx_train]
            X_test  = X[idx_test];  y_test  = labels[idx_test]

            scaler = StandardScaler() if args.standardize else None
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

            t1 = time.time()
            clf = LogisticRegression(max_iter=args.dw_max_iter, solver=args.dw_solver)
            clf.fit(X_train, y_train)
            t_train = time.time() - t1 if args.time_each_step else 0.0

            t2 = time.time()
            y_pred_train = clf.predict(X_train)
            y_pred_test  = clf.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test  = accuracy_score(y_test,  y_pred_test)
            t_eval = time.time() - t2 if args.time_each_step else 0.0

            rows.append({
                "seed": seed, "variant": "dropWL-mean",
                "acc_train": acc_train, "acc_test": acc_test,
                "time_repr": t_repr, "time_train": t_train, "time_eval": t_eval,
                "time_total": t_repr + t_train + t_eval,
                "p": args.dw_p, "R": args.dw_R, "t": args.dw_t, "kmax": args.dw_kmax
            })

            # Learning curve
            if args.learning_curves:
                lc = learning_curve_variant(
                    "dropWL-mean", graphs, labels, idx_train, idx_test,
                    {"p": args.dw_p, "R": args.dw_R, "t": args.dw_t, "kmax": args.dw_kmax,
                     "max_iter": args.dw_max_iter, "solver": args.dw_solver},
                    standardize=args.standardize, seed=seed
                )
                lc_data["dropWL-mean"].append(lc)

        # ===== Variante 3: dropWL+MLP =====
        if args.run_drop_mlp:
            # import torch aquí para no romper ARM si no se usa
            import torch  # noqa: F401
            res = train_dropwl_mlp(
                graphs, labels, idx_train, idx_test,
                p=args.mlp_p, R=args.mlp_R, t=args.mlp_t, kmax=args.mlp_kmax,
                layers=args.mlp_layers, hidden=args.mlp_hidden, d=args.mlp_d, activation=args.mlp_act,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                early_stop_patience=args.early_stop_patience, seed=seed
            )
            rows.append({
                "seed": seed, "variant": "dropWL+MLP",
                "acc_train": res["acc_train"], "acc_test": res["acc_test"],
                "time_repr": res["time_repr"], "time_train": res["time_train"], "time_eval": res["time_eval"],
                "time_total": res["time_total"],
                "p": args.mlp_p, "R": args.mlp_R, "t": args.mlp_t, "kmax": args.mlp_kmax,
                "layers": args.mlp_layers, "hidden": args.mlp_hidden, "d": args.mlp_d, "act": args.mlp_act,
                "epochs": args.epochs, "lr": args.lr, "wd": args.weight_decay
            })

            # Learning curve
            if args.learning_curves:
                lc = learning_curve_variant(
                    "dropWL+MLP", graphs, labels, idx_train, idx_test,
                    {"p": args.mlp_p, "R": args.mlp_R, "t": args.mlp_t, "kmax": args.mlp_kmax,
                     "layers": args.mlp_layers, "hidden": args.mlp_hidden, "d": args.mlp_d, "act": args.mlp_act,
                     "epochs": args.epochs, "lr": args.lr, "wd": args.weight_decay,
                     "patience": args.early_stop_patience},
                    standardize=False, seed=seed
                )
                lc_data["dropWL+MLP"].append(lc)

    # ------- Guardar resultados
    df = pd.DataFrame(rows)
    df.to_csv(outdir/"results.csv", index=False)

    # ------- Resúmenes + pruebas apareadas (WL vs drop)
    tables_dir = outdir/"tables"
    tables_dir.mkdir(exist_ok=True, parents=True)

    # Resumen por variante
    summ = df.groupby("variant")["acc_test"].agg(["mean","std","count"]).reset_index()
    summ.to_csv(tables_dir/"acc_by_variant.csv", index=False)

    # Pruebas apareadas por semilla (Δ = drop - WL)
    paired_rows = []
    variants_to_compare = [v for v in ["dropWL-mean","dropWL+MLP"] if v in df["variant"].unique()]
    for var in variants_to_compare:
        deltas = []
        common_seeds = sorted(set(df[df["variant"]=="WL"]["seed"]) & set(df[df["variant"]==var]["seed"]))
        for s in common_seeds:
            wl_acc = float(df[(df["variant"]=="WL") & (df["seed"]==s)]["acc_test"].mean())
            var_acc = float(df[(df["variant"]==var) & (df["seed"]==s)]["acc_test"].mean())
            deltas.append(var_acc - wl_acc)
        deltas = np.array(deltas, dtype=float)
        m, lo, hi = bootstrap_ci_mean_delta(deltas)
        # p-valor de permutación apareada
        x = np.array([float(df[(df["variant"]==var)&(df["seed"]==s)]["acc_test"].mean()) for s in common_seeds])
        y = np.array([float(df[(df["variant"]=="WL") &(df["seed"]==s)]["acc_test"].mean()) for s in common_seeds])
        pval = paired_permutation_test(x, y, n_perm=10000)
        paired_rows.append({"compare": f"{var} vs WL", "delta_mean": m, "ci_lo": lo, "ci_hi": hi, "p_perm_two_sided": pval, "n_pairs": len(common_seeds)})
    pd.DataFrame(paired_rows).to_csv(tables_dir/"paired_tests.csv", index=False)

    # ------- Figuras y PDF
    figs_dir = outdir/"figures"
    figs_dir.mkdir(exist_ok=True, parents=True)
    pdf = PdfPages(figs_dir/"report_compare.pdf")

    def savefig(name):
        plt.tight_layout()
        plt.savefig(figs_dir/name, dpi=200, bbox_inches="tight")
        pdf.savefig(bbox_inches="tight")
        plt.close()

    # Barras: media por variante
    plt.figure(figsize=(6,4))
    x = np.arange(len(summ))
    plt.bar(x, summ["mean"].values)
    plt.xticks(x, summ["variant"].tolist())
    for i,(m,s,c) in enumerate(zip(summ["mean"], summ["std"], summ["count"])):
        plt.text(i, m, f"{m:.3f}\n±{(0 if pd.isna(s) else s):.3f}\n(n={int(c)})", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Accuracy test (media)")
    plt.title("Precisión promedio por variante")
    savefig("01_bar_acc_by_variant.png")

    # Boxplot: distribución por variante
    plt.figure(figsize=(6,4))
    data_box = [df[df["variant"]==v]["acc_test"].values for v in summ["variant"].tolist()]
    plt.boxplot(data_box, labels=summ["variant"].tolist())
    plt.ylabel("Accuracy test")
    plt.title("Distribución de accuracy por variante")
    savefig("02_box_acc_by_variant.png")

    # Precisión vs costo (puntos por seed)
    plt.figure(figsize=(6,4))
    for v in summ["variant"].tolist():
        sub = df[df["variant"]==v]
        plt.scatter(sub["time_total"].values, sub["acc_test"].values, label=v)
    plt.xlabel("Tiempo total (s)")
    plt.ylabel("Accuracy test")
    plt.title("Precisión vs Costo")
    plt.legend()
    savefig("03_acc_vs_time.png")

    # Curvas de aprendizaje
    if args.learning_curves:
        # promediar por variante y seed
        def mean_lc(list_lc):
            # Aseguramos mismas fracciones
            if not list_lc:
                return None
            fracs = list_lc[0]["fracs"]
            M = []
            for lc in list_lc:
                M.append(lc["acc_test"])
            M = np.stack(M, axis=0)  # (n_seeds, n_fracs)
            return fracs, M.mean(axis=0), M.std(axis=0)

        for variant in ["WL","dropWL-mean","dropWL+MLP"]:
            lst = lc_data[variant]
            if not lst:
                continue
            fracs, mu, sd = mean_lc(lst)
            plt.figure(figsize=(6,4))
            plt.plot(fracs, mu, marker="o")
            plt.fill_between(fracs, mu - sd, mu + sd, alpha=0.2)
            plt.xlabel("Fracción del set de entrenamiento")
            plt.ylabel("Accuracy test (promedio ± DE)")
            plt.title(f"Curva de aprendizaje — {variant}")
            savefig(f"04_learning_curve_{variant.replace('+','_').replace('-','_')}.png")

        # Comparativa en un mismo plot (si hay varias)
        plt.figure(figsize=(6,4))
        for variant in ["WL","dropWL-mean","dropWL+MLP"]:
            lst = lc_data[variant]
            if not lst:
                continue
            fracs, mu, sd = mean_lc(lst)
            plt.plot(fracs, mu, marker="o", label=variant)
        plt.xlabel("Fracción del set de entrenamiento")
        plt.ylabel("Accuracy test (promedio)")
        plt.title("Curvas de aprendizaje — Comparación")
        plt.legend()
        savefig("05_learning_curves_compare.png")

    pdf.close()

    # Resumen en texto
    summary = {
        "seeds": args.seeds,
        "variants": list(df["variant"].unique()),
        "mean_acc_by_variant": {row["variant"]: float(row["mean"]) for _, row in summ.iterrows()}
    }
    with open(outdir/"tables"/"SUMMARY.json","w") as f:
        json.dump(summary, f, indent=2)

    print("Listo. Resultados en:", str(outdir))
    print("CSV principal:", str(outdir/"results.csv"))
    print("Figuras + PDF:", str(outdir/"figures"/"report_compare.pdf"))
    print("Tablas:", str(outdir/"tables"))

if __name__ == "__main__":
    main()
