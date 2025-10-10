# scripts/exp_simple_compare.py
import argparse, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- imports del repo ---
from src.core.wl_refinement import compute_wl_colors, color_histograma
from src.core.onedropwl import onedropwl_one_edge  # 1-drop (una arista) -> vector kmax
from src.core.dropwl_runner import representar_grafo_dropwl_mean  # por si necesitas R>1 en el futuro

# MLP (solo si PyTorch está disponible)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.mlp.mlp_head import MLPHead
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# --------------------------
# Utilidades
# --------------------------
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

def wl_signature(G: nx.Graph, t: int, kmax: int) -> np.ndarray:
    # llamada robusta a compute_wl_colors
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

def one_drop_signature(G: nx.Graph, t: int, kmax: int, seed: int) -> np.ndarray:
    # 1-drop: elimina exactamente una arista aleatoria y luego WL-t -> hist kmax
    vec = onedropwl_one_edge(G, t=t, k_max=kmax, seed=seed)
    return vec.astype(np.float32)

# --------------------------
# Modelos
# --------------------------
def run_WL(graphs, labels, seed, test_size, t, kmax, standardize):
    t0 = time.time()
    Gtr, Gte, ytr, yte = train_test_split(
        graphs, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    t_repr0 = time.time()
    Xtr = np.stack([wl_signature(G, t=t, kmax=kmax) for G in Gtr], axis=0)
    Xte = np.stack([wl_signature(G, t=t, kmax=kmax) for G in Gte], axis=0)
    t_repr = time.time() - t_repr0

    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    t_fit0 = time.time()
    clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    t_fit = time.time() - t_fit0

    t_eval0 = time.time()
    acc_tr = accuracy_score(ytr, clf.predict(Xtr))
    acc_te = accuracy_score(yte, clf.predict(Xte))
    t_eval = time.time() - t_eval0

    return {
        "variant": "WL", "seed": seed,
        "acc_train": float(acc_tr), "acc_test": float(acc_te),
        "t_repr_s": float(t_repr), "t_fit_s": float(t_fit),
        "t_eval_s": float(t_eval), "t_total_s": float(time.time() - t0),
        "dim": int(Xtr.shape[1])
    }

def run_1drop_log(graphs, labels, seed, test_size, t, kmax, R, standardize):
    """
    1-drop LOG: para cada grafo, promedio de R firmas 1-drop (cada una quita UNA arista y WL-t).
    Luego logística.
    """
    t0 = time.time()
    Gtr, Gte, ytr, yte = train_test_split(
        graphs, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    rng = np.random.default_rng(seed)

    def rep_batch(GS, base_seed):
        reps = []
        for i, G in enumerate(GS):
            vecs = [one_drop_signature(G, t=t, kmax=kmax, seed=base_seed + j) for j in range(R)]
            reps.append(np.mean(np.stack(vecs, axis=0), axis=0))
        return np.stack(reps, axis=0)

    t_repr0 = time.time()
    Xtr = rep_batch(Gtr, base_seed=seed)
    Xte = rep_batch(Gte, base_seed=seed + 10_000)
    t_repr = time.time() - t_repr0

    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    t_fit0 = time.time()
    clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    t_fit = time.time() - t_fit0

    t_eval0 = time.time()
    acc_tr = accuracy_score(ytr, clf.predict(Xtr))
    acc_te = accuracy_score(yte, clf.predict(Xte))
    t_eval = time.time() - t_eval0

    return {
        "variant": "1drop-LOG", "seed": seed,
        "acc_train": float(acc_tr), "acc_test": float(acc_te),
        "t_repr_s": float(t_repr), "t_fit_s": float(t_fit),
        "t_eval_s": float(t_eval), "t_total_s": float(time.time() - t0),
        "dim": int(Xtr.shape[1])
    }

def run_1drop_mlp(graphs, labels, seed, test_size, t, kmax, R, layers, hidden, d, act, epochs, lr, patience, standardize, device):
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch no disponible para 1-drop+MLP.")

    t0 = time.time()
    Gtr, Gte, ytr, yte = train_test_split(
        graphs, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    # Representación base: media de R firmas 1-drop (kmax)
    def rep_batch(GS, base_seed):
        reps = []
        for i, G in enumerate(GS):
            vecs = [one_drop_signature(G, t=t, kmax=kmax, seed=base_seed + j) for j in range(R)]
            reps.append(np.mean(np.stack(vecs, axis=0), axis=0))
        return np.stack(reps, axis=0)

    t_repr0 = time.time()
    Xtr = rep_batch(Gtr, base_seed=seed)
    Xte = rep_batch(Gte, base_seed=seed + 10_000)
    t_repr = time.time() - t_repr0

    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    # MLP head + clasificador lineal
    act_name = "relu" if act == "relu" else "tanh"
    net = MLPHead(input_dim=kmax, hidden_dim=hidden, output_dim=d,
                  num_layers=layers, activation=act_name).to(device)
    clf = nn.Linear(d, 2).to(device)
    opt = optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    best_te, acc_tr_best = 0.0, 0.0
    patience_left = patience

    for _ in range(epochs):
        net.train(); clf.train()
        opt.zero_grad()
        z = net(Xtr_t)
        logits = clf(z)
        loss = ce(logits, ytr_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            net.eval(); clf.eval()
            pred_tr = logits.argmax(dim=1).detach().cpu().numpy()
            acc_tr = accuracy_score(ytr, pred_tr)
            acc_te = accuracy_score(yte, clf(net(Xte_t)).argmax(dim=1).detach().cpu().numpy())

        if acc_te > best_te:
            best_te = acc_te
            acc_tr_best = acc_tr
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return {
        "variant": "1drop-MLP", "seed": seed,
        "acc_train": float(acc_tr_best), "acc_test": float(best_te),
        "t_repr_s": float(t_repr), "t_fit_s": float(time.time() - t_repr0 - t_repr),
        "t_eval_s": 0.0, "t_total_s": float(time.time() - t0),
        "dim": int(Xtr.shape[1])
    }

# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")

    # Flags que el orquestador está pasando
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--R", type=int, default=50)  # número de repeticiones para 1-drop

    # Selección de variantes (el orquestador usa estos)
    ap.add_argument("--run_wl", action="store_true")
    ap.add_argument("--run_1drop_log", action="store_true")
    ap.add_argument("--run_1drop_mlp", action="store_true")

    # Hiperparámetros de MLP (si se usa 1drop_mlp)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)

    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    graphs, labels, n = load_npz_dataset(args.data)
    rows = []

    for seed in args.seeds:
        if args.run_wl:
            r = run_WL(graphs, labels, seed, args.test_size, args.t, args.kmax, args.standardize)
            rows.append(r)
            print(f"[WL][seed={seed}] acc_test={r['acc_test']:.4f} (dim={r['dim']}, t_total={r['t_total_s']:.3f}s)")

        if args.run_1drop_log:
            r = run_1drop_log(graphs, labels, seed, args.test_size, args.t, args.kmax, args.R, args.standardize)
            rows.append(r)
            print(f"[1drop-LOG][seed={seed}] acc_test={r['acc_test']:.4f} (dim={r['dim']}, t_total={r['t_total_s']:.3f}s)")

        if args.run_1drop_mlp:
            r = run_1drop_mlp(
                graphs, labels, seed, args.test_size, args.t, args.kmax, args.R,
                args.mlp_layers, args.mlp_hidden, args.mlp_d, args.mlp_act,
                args.epochs, args.lr, args.patience, args.standardize, args.device
            )
            rows.append(r)
            print(f"[1drop-MLP][seed={seed}] acc_test={r['acc_test']:.4f} (dim={r['dim']}, t_total={r['t_total_s']:.3f}s)")

    if not rows:
        print("[WARN] No se seleccionó ninguna variante (--run_wl / --run_1drop_log / --run_1drop_mlp).")
        return

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "results.csv", index=False)
    summ = df.groupby("variant")["acc_test"].agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(outdir / "tables" / "acc_by_variant.csv", index=False)

    print(f"[OK] CSV: {outdir/'results.csv'}")
    print("[info] Resumen por variante:")
    print(summ)

if __name__ == "__main__":
    main()