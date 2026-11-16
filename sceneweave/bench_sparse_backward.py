"""Micro-benchmark: dense vs sparse tile backward at 1200x680, 5k and 20k splats, on a converged-ish scene.

    python sceneweave/bench_sparse_backward.py [--reps 7] [--out result.json]

Per (splats, arm) and repetition: a fresh trainer is trained `--pretrain` dense steps (the converged-ish
start; smooth image with sharp patches), then `--steps` more steps in batches of 4 with the arm's settings
(batches start on multiples of 4, so with sparse_full_every=4 every batch has one dense sentinel step and
three sparse steps), timed per batch. Then `--finish` further steps with the same settings and the PSNR
over all training views. Reported: median over repetitions of ms/step, tiles dispatched fraction
(backward_tiles_dispatched / backward_tiles_total over the timed steps) and PSNR.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import append_scenario as sc

ARMS = {"dense": 0.0, "sparse_q0.50": 0.5, "sparse_q0.75": 0.75}


def psnr(mod) -> float:
    errs = []
    for slot in range(int(mod.num_train)):
        mod.render_train(slot)
        img = np.asarray(mod.get_train_image(slot), dtype=np.float32)
        h, w = img.shape[:2]
        pred = np.clip(np.asarray(mod.pixel_state, dtype=np.float32).reshape(h, w, 4)[..., :3], 0.0, 1.0)
        errs.append(float(((pred - img[..., :3] / 255.0) ** 2).mean()))
    return float(-10.0 * np.log10(np.mean(errs)))


def one_rep(ds: Path, out: Path, quantile: float, pretrain: int, steps: int, finish: int, halo: int):
    mod = sc.new_trainer(ds, out, sparse_tile_quantile=0.0)
    n = int(mod.num_train)
    step = 0
    while step < pretrain:  # converged-ish start, dense
        mod.train_steps([(step + i) % n for i in range(4)], step)
        step += 4
    mod.set_sparse_backward(quantile, halo, 4)
    c0 = mod.get_work_counters()
    per_step = []
    for _ in range(steps // 4):
        t0 = time.perf_counter()
        mod.train_steps([(step + i) % n for i in range(4)], step)
        per_step.append((time.perf_counter() - t0) * 1e3 / 4)
        step += 4
    c1 = mod.get_work_counters()
    frac = (c1["backward_tiles_dispatched"] - c0["backward_tiles_dispatched"]) / (
        c1["backward_tiles_total"] - c0["backward_tiles_total"])
    for _ in range(finish // 4):
        mod.train_steps([(step + i) % n for i in range(4)], step)
        step += 4
    return statistics.median(per_step), frac, psnr(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--splats", type=int, nargs="+", default=[5000, 20000])
    ap.add_argument("--pretrain", type=int, default=200)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--finish", type=int, default=100)
    ap.add_argument("--halo", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = {"build": str(sc.ROOT), "reps": args.reps, "image": "1200x680", "halo": args.halo, "cells": {}}
    with tempfile.TemporaryDirectory() as td:
        for n_splats in args.splats:
            ds = sc.build_dataset(Path(td) / f"ds{n_splats}", n_images=6, n_points=n_splats, w=1200, h=680,
                                  structured=True)
            for arm, q in ARMS.items():
                runs = [one_rep(ds, Path(td) / f"o{n_splats}_{arm}_{r}", q, args.pretrain, args.steps, args.finish,
                                args.halo) for r in range(args.reps)]
                result["cells"][f"{n_splats}_splats_{arm}"] = {
                    "ms_per_step_median": round(statistics.median(r[0] for r in runs), 3),
                    "ms_per_step_all": [round(r[0], 3) for r in runs],
                    "tiles_dispatched_fraction": round(statistics.median(r[1] for r in runs), 3),
                    "psnr_median": round(float(statistics.median(r[2] for r in runs)), 3),
                }
                print(n_splats, arm, result["cells"][f"{n_splats}_splats_{arm}"], flush=True)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
