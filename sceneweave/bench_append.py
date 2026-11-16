"""Micro-benchmark: 50 appends of 120 splats onto a 5k-splat model with one train step between appends.

    python sceneweave/bench_append.py [--reps 5]            # this build
    VKSPLAT_PY_DIR=<baseline dir> python sceneweave/bench_append.py   # another build (e.g. fd1a50d)

Reports the median per-call append ms, train-step ms and their per-cycle sum (the unfenced append defers its
GPU copy into the following step, so the sum is the fair comparison), over `--reps` fresh trainers.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import append_scenario as sc


def one_rep(ds: Path, out: Path, appends: int, m: int):
    mod = sc.new_trainer(ds, out)
    n_train = int(mod.num_train)
    rng = np.random.default_rng(3)
    batches = [sc.make_batch(rng, m, i % 8) for i in range(appends)]
    for i in range(3):  # warm-up: pipelines, descriptors, first-touch buffers
        mod.train_step(i % n_train, i)
    c0 = dict(mod.get_work_counters())
    app, step = [], []
    for i, b in enumerate(batches):
        t0 = time.perf_counter()
        mod.append_splats_preserving_optimizer(*b)
        t1 = time.perf_counter()
        mod.train_step(i % n_train, 3 + i)
        t2 = time.perf_counter()
        app.append((t1 - t0) * 1e3)
        step.append((t2 - t1) * 1e3)
    c1 = dict(mod.get_work_counters())
    return app, step, {k: c1[k] - c0.get(k, 0) for k in c1 if k.startswith(("append_", "step_buffer", "fence_waits"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--appends", type=int, default=50)
    ap.add_argument("--m", type=int, default=120)
    ap.add_argument("--points", type=int, default=5000)
    args = ap.parse_args()
    meds = {"append_ms": [], "step_ms": [], "cycle_ms": []}
    counters = None
    with tempfile.TemporaryDirectory() as td:
        ds = sc.build_dataset(Path(td) / "ds", n_images=6, n_points=args.points, w=256, h=192)
        for r in range(args.reps):
            app, step, counters = one_rep(ds, Path(td) / f"o{r}", args.appends, args.m)
            meds["append_ms"].append(statistics.median(app))
            meds["step_ms"].append(statistics.median(step))
            meds["cycle_ms"].append(statistics.median(a + s for a, s in zip(app, step)))
    print(json.dumps({"reps": args.reps, "build": str(sc.ROOT),
                      **{k: {"median": round(statistics.median(v), 3), "all": [round(x, 3) for x in v]} for k, v in meds.items()},
                      "counters_last_rep": counters}, indent=1))


if __name__ == "__main__":
    main()
