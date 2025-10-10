# scripts/run_phase1_simple.py
import argparse, os, shlex, subprocess, sys, time
from pathlib import Path

DATASETS = {
    # Clasificación de grafos
    "C4_n8":   "data_paper/c4_n8_p030_bal.npz",
    "C4_n16":  "data_paper/c4_n16_p030_bal.npz",
    "C4_n24":  "data_paper/c4_n24_p030_bal.npz",
    "C4_n32":  "data_paper/c4_n32_p030_bal.npz",
    "C4_n40":  "data_paper/c4_n40_p030_bal.npz",
    "C4_n44":  "data_paper/c4_n44_p030_bal.npz",
    "LIMITS1": "data_synth/limits/limits1_s20250925.npz",
    "LIMITS2": "data_synth/limits/limits2_s20250925.npz",
    "SKIP":    "data_synth/skip/skip_n32_s20250925.npz",

    # Clasificación de nodos
    "LCC_nodes":       "data_synth/lcc/lcc_nodes_n32_s20250925.npz",
    "TRIANGLES_nodes": "data_synth/triangles/tri_nodes_n32_s20250925.npz",
}

MODELOS = [
    ("WL",            "--run_wl"),
    ("1drop_log",     "--run_1drop_log"),   # 1-dropWL con clasificador logístico
    ("1drop_mlp",     "--run_1drop_mlp"),   # 1-dropWL con MLP integrado
]


def run(cmd: str) -> int:
    """Ejecuta un comando mostrando stdout/stderr y fijando PYTHONPATH correctamente."""
    import os, subprocess, shlex, sys
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    return subprocess.run(
        shlex.split(cmd),
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env
    ).returncode


def main():
    ap = argparse.ArgumentParser(description="Fase 1 (modelos simples, sin p): WL, 1-dropWL-LOG, 1-dropWL-MLP")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250925, 20250926, 20250927, 20250928, 20250929])
    ap.add_argument("--outroot", type=str, default="results/fase1_simple")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")

    # Hiperparámetros (mantener consistentes con exp_simple_compare.py)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax_graph", type=int, default=40)  # para grafos (se ajusta si n<kmax)
    ap.add_argument("--kmax_node", type=int, default=40)   # para node-level

    # 1-dropWL (LOG y MLP): rondas R fijas = 50
    ap.add_argument("--R", type=int, default=50)

    # MLP (sólo para 1drop_mlp)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu")

    args = ap.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    # Clasificamos datasets por tipo de tarea
    graph_datasets = {k: v for k, v in DATASETS.items() if not k.endswith("_nodes")}
    node_datasets  = {k: v for k, v in DATASETS.items() if k.endswith("_nodes")}

    jobs = []
    for name, path in DATASETS.items():
        task = "node" if name.endswith("_nodes") else "graph"
        for seed in args.seeds:
            for model_name, model_flag in MODELOS:
                # Directorio por dataset/modelo/seed
                outdir = outroot / name / model_name / f"seed_{seed}"
                outdir.mkdir(parents=True, exist_ok=True)

                # Ajustes de kmax (si es grafos y n < kmax, exp_simple_compare debe truncar o nosotros fijar kmax=n)
                # Aquí pasamos kmax según tipo:
                kmax = args.kmax_node if task == "node" else args.kmax_graph

                # Construimos comando base
                cmd = f"python scripts/exp_simple_compare.py " \
                      f"--data {path} --seeds {seed} --test_size {args.test_size} " \
                      f"--t {args.t} --kmax {kmax} --R {args.R} {model_flag} " \
                      f"--mlp_layers {args.mlp_layers} --mlp_hidden {args.mlp_hidden} " \
                      f"--mlp_d {args.mlp_d} --mlp_act {args.mlp_act} " \
                      f"--device {args.device} --outdir {outdir}"
                if args.standardize:
                    cmd += " --standardize"

                jobs.append((name, seed, model_name, path, str(outdir), cmd))

    K = len(jobs)
    print(f"[PLAN] Total de trabajos: {K}")
    t0 = time.time()

    for i, (dsname, seed, model_name, path, outdir, cmd) in enumerate(jobs, start=1):
        print(f"[{i}/{K}  {100*i/K:5.1f}%] DATASET={dsname:14s}  MODEL={model_name:11s}  SEED={seed}  -> {outdir}")
        rc = run(cmd)
        if rc != 0:
            print(f"[WARN] job falló con rc={rc}: {cmd}")

    print(f"[DONE] Tiempo total: {time.time()-t0:.1f}s")
    print(f"[OUT] Raíz de resultados: {outroot}")

if __name__ == "__main__":
    main()
