"""Micro-benchmark: raster-backward gradient skip for frozen splats (IMAGINED / protected LOCKED groups).

    python sceneweave/bench_freeze_backward.py [--reps 7] [--variant 3] [--out result.json]

20k splats in 100 groups at 1200x680; a fraction of the groups is frozen. Per (state, fraction, arm) and
repetition a fresh trainer runs `--pretrain` steps, then `--steps` timed steps in batches of 4. Arms: `skip`
(freeze_skip_backward on) and `full` (atomics of the frozen splats still run). Reported: median ms/step over
repetitions, with the raster backward pinned to one variant so that both arms run the same kernel.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import append_scenario as sc

N_GROUPS, FIRST_GID = 100, 10


def one_rep(ds, out, n_splats, variant, state, fraction, skip, pretrain, steps, args_scale):
    mod = sc.new_trainer(ds, out, mcmc_protect_structural_groups=True, sparse_tile_quantile=0.0)
    mod.force_backward_variant(variant)
    mod.set_freeze_skip_backward(skip)
    rng = np.random.default_rng(7)
    xyz, sh, rot, scales, op, _ = sc.make_batch(rng, n_splats, 0)
    scales *= args_scale
    gids = rng.integers(FIRST_GID, FIRST_GID + N_GROUPS, size=n_splats).astype(np.int32)
    mod.append_splats_preserving_optimizer(xyz, sh, rot, scales, op, gids)
    k = int(round(fraction * N_GROUPS))
    if k:
        mod.set_group_states(np.arange(FIRST_GID, FIRST_GID + k, dtype=np.int32), np.full(k, state, dtype=np.uint32))
    n = int(mod.num_train)
    step = 0
    while step < pretrain:
        mod.train_steps([(step + i) % n for i in range(4)], step)
        step += 4
    per_step = []
    for _ in range(steps // 4):
        t0 = time.perf_counter()
        mod.train_steps([(step + i) % n for i in range(4)], step)
        per_step.append((time.perf_counter() - t0) * 1e3 / 4)
        step += 4
    return statistics.median(per_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--splats", type=int, default=20000)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier of the synthetic splat scales (0.01-0.05)")
    ap.add_argument("--variant", type=int, default=3)
    ap.add_argument("--pretrain", type=int, default=16)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = {"build": str(sc.ROOT), "reps": args.reps, "image": "1200x680", "splats": args.splats,
              "variant": args.variant, "cells": {}}
    cells = [("none", 0, 0.0, False)] + [(name, state, f, skip) for name, state in (("imagined", 1), ("locked", 2))
                                         for f in (0.3, 0.6) for skip in (False, True)]
    with tempfile.TemporaryDirectory() as td:
        ds = sc.build_dataset(Path(td) / "ds", n_images=6, n_points=100, w=1200, h=680, structured=True)
        for name, state, f, skip in cells:
            runs = [one_rep(ds, Path(td) / f"o{r}", args.splats, args.variant, state, f, skip, args.pretrain, args.steps, args.scale)
                    for r in range(args.reps)]
            key = f"{name}_{int(f * 100)}pct_{'skip' if skip else 'full'}"
            result["cells"][key] = {"ms_per_step_median": round(statistics.median(runs), 3),
                                    "ms_per_step_all": [round(x, 3) for x in runs]}
            print(key, result["cells"][key], flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
