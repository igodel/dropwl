# scripts/check_phase1_datasets.py
from pathlib import Path
import numpy as np

def check_graph_npz(path):
    if not path.exists(): return f"[FALTA] {path}"
    try:
        d = np.load(path, allow_pickle=True)
        if not all(k in d for k in ["edges_list", "labels", "n"]):
            return f"[ERROR] {path} -> faltan llaves (edges_list/labels/n)"
        N = len(d["edges_list"]); n = int(d["n"]); labs = d["labels"]
        return f"[OK] {path} -> N={N}, n={n}, clases={np.unique(labs, return_counts=True)}"
    except Exception as e:
        return f"[ERROR] {path} -> {repr(e)}"

def check_node_npz(path):
    if not path.exists(): return f"[FALTA] {path}"
    try:
        d = np.load(path, allow_pickle=True)
        if not all(k in d for k in ["edges_list", "node_labels_list", "n"]):
            return f"[ERROR] {path} -> faltan llaves (edges_list/node_labels_list/n)"
        N = len(d["edges_list"]); n = int(d["n"]); M = len(d["node_labels_list"])
        shapes_ok = all(len(lbl) == n for lbl in d["node_labels_list"])
        return f"[OK] {path} -> N={N}, n={n}, node_labels={M}, shapes_ok={shapes_ok}"
    except Exception as e:
        return f"[ERROR] {path} -> {repr(e)}"

def main():
    paths = [
        # Graph classification que ya tienes:
        Path("data_paper/c4_n8_p030_bal.npz"),
        Path("data_paper/c4_n16_p030_bal.npz"),
        Path("data_paper/c4_n24_p030_bal.npz"),
        Path("data_paper/c4_n32_p030_bal.npz"),
        Path("data_paper/c4_n40_p030_bal.npz"),
        Path("data_paper/c4_n44_p030_bal.npz"),
        Path("data_synth/limits/limits1_s20250925.npz"),
        Path("data_synth/limits/limits2_s20250925.npz"),
        # Faltantes (SKIP-CIRCLES graph-level, LCC y TRIANGLES node-level):
        Path("data_synth/skip/skip_n32_s20250925.npz"),
        Path("data_synth/lcc/lcc_nodes_n32_s20250925.npz"),
        Path("data_synth/triangles/tri_nodes_n32_s20250925.npz"),
    ]
    for p in paths:
        if "lcc" in str(p) or "tri_nodes" in str(p):
            print(check_node_npz(p))
        elif "skip" in str(p) or "c4_" in str(p) or "limits" in str(p):
            print(check_graph_npz(p))
        else:
            print(f"[WARN] Sin regla para {p}")

if __name__ == "__main__":
    main()
