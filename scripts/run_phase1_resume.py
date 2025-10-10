#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orquestador robusto para Fase 1 (WL, 1drop-LOG, 1drop-MLP):
- Reanuda automáticamente SOLO lo que falta (idempotente).
- Sin buffering (PYTHONUNBUFFERED=1) y log por dataset (run.log).
- Timeout por dataset (default 3600s) y hasta 2 reintentos automáticos.
- % de avance y ETA simple.
- Detiene limpiamente y puedes relanzar: reanudará pendientes.

Uso recomendado:
  caffeinate -dimsu python -u scripts/run_phase1_resume.py \
    --outroot results/fase1_simple_all \
    --test_size 0.30 \
    --seeds 20250925 20250926 20250927 20250928 20250929 \
    --device cpu --standardize \
    --t 3 --kmax 40 --R 50 \
    --mlp_layers 2 --mlp_hidden 128 --mlp_d 64 --mlp_act relu \
    --timeout 3600
"""
import argparse, os, sys, shlex, time, datetime, subprocess
from pathlib import Path

DATA = {
    "C4_n8":  "data_paper/c4_n8_p030_bal.npz",
    "C4_n16": "data_paper/c4_n16_p030_bal.npz",
    "C4_n24": "data_paper/c4_n24_p030_bal.npz",
    "C4_n32": "data_paper/c4_n32_p030_bal.npz",
    "C4_n40": "data_paper/c4_n40_p030_bal.npz",
    "C4_n44": "data_paper/c4_n44_p030_bal.npz",
    "LIMITS1": "data_synth/limits/limits1_s20250925.npz",
    "LIMITS2": "data_synth/limits/limits2_s20250925.npz",
    "SKIP32":  "data_synth/skip/skip_n32_s20250925.npz",
    "LCC32":   "data_synth/lcc/lcc_nodes_n32_s20250925.npz",
    "TRI32":   "data_synth/triangles/tri_nodes_n32_s20250925.npz",
}

def exists_ok(outdir: Path) -> bool:
    return (outdir / "results.csv").exists()

def run_unbuffered(cmd: str, log_file: Path, env: dict, timeout: int) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", buffering=1) as lf:
        lf.write(f"\n[{datetime.datetime.now()}] CMD: {cmd}\n")
        lf.flush()
        p = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1
        )
        start = time.time()
        rc = None
        try:
            for line in iter(p.stdout.readline, ''):
                lf.write(line)
                lf.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                if timeout and (time.time() - start) > timeout:
                    p.kill()
                    rc = 124  # timeout
                    break
            if rc is None:
                rc = p.wait()
        except Exception as e:
            try:
                p.kill()
            except Exception:
                pass
            lf.write(f"[EXC] {repr(e)}\n")
            rc = 1
        lf.write(f"[END] rc={rc} elapsed={time.time()-start:.1f}s\n")
        lf.flush()
        return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", type=str, default="results/fase1_simple_all")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_d", type=int, default=64)
    ap.add_argument("--mlp_act", type=str, default="relu")
    ap.add_argument("--timeout", type=int, default=3600)   # 1h por dataset
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    # Construye lista de datasets pendientes (idempotente).
    datasets = list(DATA.keys())
    pending = [nm for nm in datasets if not exists_ok(outroot / nm)]
    done    = [nm for nm in datasets if nm not in pending]

    total = len(pending)
    if total == 0:
        print("[OK] Nada pendiente. Todo completo.")
        return

    print(f"[PLAN] Pendientes: {total} -> {pending}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "."

    # Reduce threads de BLAS para evitar cuelgues por saturación con Rosetta
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    started = time.time()
    for i, ds in enumerate(pending, start=1):
        data = DATA[ds]
        outdir = outroot / ds
        outdir.mkdir(parents=True, exist_ok=True)
        log = outdir / "run.log"

        # Lanza los 3 modelos en una sola invocación (como ya venías usando)
        cmd = (
            f"python -u scripts/exp_simple_compare.py "
            f"--data {data} --seeds {' '.join(map(str,args.seeds))} "
            f"--test_size {args.test_size} --t {args.t} --kmax {args.kmax} --R {args.R} "
            f"--run_wl --run_1drop_log --run_1drop_mlp "
            f"--mlp_layers {args.mlp_layers} --mlp_hidden {args.mlp_hidden} "
            f"--mlp_d {args.mlp_d} --mlp_act {args.mlp_act} "
            f"--epochs 30 --lr 1e-3 --patience 8 --device {args.device} "
            f"--outdir {outdir} {'--standardize' if args.standardize else ''}"
        )

        # Reintentos con timeout
        attempt = 0
        while attempt <= args.max_retries:
            attempt += 1
            pct = (i-1)/total*100
            eta = "?"
            if i > 1:
                elapsed = time.time() - started
                eta = f"{(elapsed/(i-1)*(total-(i-1)))/60:.1f} min"
            print(f"[{i}/{total}  {pct:4.1f}%] DATASET={ds}  intento={attempt}  ETA~{eta}")
            rc = run_unbuffered(cmd, log, env, args.timeout)
            if rc == 0 and exists_ok(outdir):
                print(f"[OK] {ds} -> {outdir/'results.csv'}")
                break
            else:
                print(f"[WARN] Falló rc={rc} (o no hay results.csv). Reintento {attempt}/{args.max_retries}…")
                time.sleep(3)
        else:
            print(f"[FATAL] {ds} agotó reintentos. Continúo con el siguiente.")

    print("[DONE] Reanudar completado. Revisa qué quedó pendiente con el verificador.")

if __name__ == "__main__":
    main()
