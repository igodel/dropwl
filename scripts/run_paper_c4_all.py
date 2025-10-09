# scripts/run_paper_c4_all.py
import argparse, shlex, subprocess, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def run(cmd: str, env=None):
    print(f"[run] {cmd}")
    res = subprocess.run(shlex.split(cmd), env=env)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def ensure_dataset(npz_path: Path, n: int, p: float, n_per_class: int, seed: int):
    if npz_path.exists():
        print(f"[ds] OK: {npz_path.name} ya existe")
        return
    cmd = (
        f"{sys.executable} {HERE/'gen_4cycle_controlled.py'} "
        f"--n {n} --p {p} --n_por_clase {n_per_class} --seed {seed} "
        f"--out {npz_path}"
    )
    run(cmd)

def main():
    ap = argparse.ArgumentParser(description="Orquestador completo: C4 (paper-like) con WL / dropWL-mean / dropWL+MLP")
    ap.add_argument("--ns", type=int, nargs="+", required=True, help="Tamaños de grafo n (ej: 8 16 24 32 40 44)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="Semillas (ej: 20250925 20250926 20250927 20250928)")
    # Dataset (paper usa p≈0.3 y balance 50/50)
    ap.add_argument("--p_edge", type=float, default=0.30)
    ap.add_argument("--n_per_class", type=int, default=600)
    ap.add_argument("--dataset_dir", type=str, default="data_paper")
    # dropWL-mean pretexto (grid)
    ap.add_argument("--dw_p_list", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--dw_R_list", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--dw_t", type=int, default=3)
    # MLP grid (capacidad + pretexto)
    ap.add_argument("--mlp_layers_grid", type=int, nargs="+", default=[2, 3, 5, 10])
    ap.add_argument("--mlp_hidden_grid", type=int, nargs="+", default=[20, 64, 128])
    ap.add_argument("--mlp_d_grid", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--mlp_act", type=str, default="relu", choices=["relu","tanh"])
    ap.add_argument("--mlp_p_list", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--mlp_R_list", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--mlp_t", type=int, default=3)
    # WL baseline
    ap.add_argument("--wl_t", type=int, default=3)
    # Estandarización
    ap.add_argument("--standardize", action="store_true")
    # Salidas
    ap.add_argument("--outroot", type=str, required=True)
    # Generar si falta
    ap.add_argument("--generate_if_missing", action="store_true")
    args = ap.parse_args()

    outroot = Path(args.outroot); outroot.mkdir(parents=True, exist_ok=True)
    dstdir = Path(args.dataset_dir); dstdir.mkdir(parents=True, exist_ok=True)

    # Asegurar que exp_c4_compare.py existe
    exp_script = HERE / "exp_c4_compare.py"
    if not exp_script.exists():
        print(f"[ERR] No encuentro {exp_script}")
        sys.exit(1)

    # Preparar ENV con PYTHONPATH=.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # 1) Generación datasets (si procede)
    for n in args.ns:
        npz_path = dstdir / f"c4_n{n}_p{int(args.p_edge*100):03d}_bal.npz"
        if args.generate_if_missing:
            ensure_dataset(npz_path, n=n, p=args.p_edge, n_per_class=args.n_per_class, seed=args.seeds[0])

    # 2) Ejecutar WL / dropWL-mean / dropWL+MLP en todos los n y seeds
    master_rows = []  # se consolidan luego en aggregate
    for n in args.ns:
        data = dstdir / f"c4_n{n}_p{int(args.p_edge*100):03d}_bal.npz"
        if not data.exists():
            print(f"[WARN] Falta dataset {data}, me lo salto")
            continue

        nroot = outroot / f"c4_n{n}"
        (nroot / "WL").mkdir(parents=True, exist_ok=True)
        (nroot / "dropWL_mean").mkdir(parents=True, exist_ok=True)
        (nroot / "dropWL_mlp").mkdir(parents=True, exist_ok=True)

        # 2.1 WL (sin grid)
        cmd = (
            f"{sys.executable} {exp_script} "
            f"--data {data} "
            f"--seeds {' '.join(map(str, args.seeds))} "
            f"--wl_t {args.wl_t} --wl_kmax {n} "
            f"{'--standardize' if args.standardize else ''} "
            f"--outdir {nroot/'WL'}"
        )
        run(cmd, env=env)

        # 2.2 dropWL-mean (grid en p y R)
        for dw_p in args.dw_p_list:
            for dw_R in args.dw_R_list:
                outdir = nroot / "dropWL_mean" / f"p{dw_p}_R{dw_R}"
                outdir.mkdir(parents=True, exist_ok=True)
                cmd = (
                    f"{sys.executable} {exp_script} "
                    f"--data {data} "
                    f"--seeds {' '.join(map(str, args.seeds))} "
                    f"--wl_t {args.wl_t} --wl_kmax {n} "
                    f"--dw_p {dw_p} --dw_R {dw_R} --dw_t {args.dw_t} --dw_kmax {n} "
                    f"{'--standardize' if args.standardize else ''} "
                    f"--outdir {outdir}"
                )
                run(cmd, env=env)

        # 2.3 dropWL+MLP (grid en p, R y capacidad)
        # Requiere torch; si falta, el exp salta MLP y deja WL y dropWL-mean.
        for mlp_p in args.mlp_p_list:
            for mlp_R in args.mlp_R_list:
                for layers in args.mlp_layers_grid:
                    for hidden in args.mlp_hidden_grid:
                        for d in args.mlp_d_grid:
                            outdir = nroot / "dropWL_mlp" / f"p{mlp_p}_R{mlp_R}_L{layers}_H{hidden}_D{d}_{args.mlp_act}"
                            outdir.mkdir(parents=True, exist_ok=True)
                            cmd = (
                                f"{sys.executable} {exp_script} "
                                f"--data {data} "
                                f"--seeds {' '.join(map(str, args.seeds))} "
                                f"--wl_t {args.wl_t} --wl_kmax {n} "
                                f"--mlp_p {mlp_p} --mlp_R {mlp_R} --mlp_t {args.mlp_t} --mlp_kmax {n} "
                                f"--mlp_layers {layers} --mlp_hidden {hidden} --mlp_d {d} --mlp_act {args.mlp_act} "
                                f"{'--standardize' if args.standardize else ''} "
                                f"--outdir {outdir}"
                            )
                            run(cmd, env=env)

    # 3) Agregador maestro
    agg = HERE / "aggregate_paper_c4.py"
    if agg.exists():
        cmd = f"{sys.executable} {agg} --root {outroot}"
        run(cmd, env=env)
    else:
        print("[WARN] No se encontró aggregate_paper_c4.py; puedes correrlo luego manualmente.")

if __name__ == "__main__":
    main()