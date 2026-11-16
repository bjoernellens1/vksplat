"""Micro-benchmark: train steps in batches of 1, 2 and 4 at 5k and 20k splats.

    python sceneweave/bench_train_steps.py [--reps 7]                    # this build
    VKSPLAT_PY_DIR=<baseline dir> python sceneweave/bench_train_steps.py  # another build (e.g. 317f6c1)

Per (splats, batch) cell: median wall ms per step over `--reps` fresh trainers (each rep: the median over its
timed batches of batch_ms / batch) and fence waits per step from the `fence_waits_total` counter.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import append_scenario as sc


def one_rep(ds: Path, out: Path, batch: int, steps: int, pin_after: int, warmup: int):
    mod = sc.new_trainer(ds, out, pin_after)
    n_train = int(mod.num_train)
    step = 0
    for _ in range(warmup):  # warm-up: pipelines, descriptors, first-touch buffers, backward-variant sampling
        mod.train_steps([step % n_train], step)
        step += 1
    c0 = mod.get_work_counters()["fence_waits_total"]
    per_step = []
    for _ in range(steps // batch):
        slots = [(step + i) % n_train for i in range(batch)]
        t0 = time.perf_counter()
        mod.train_steps(slots, step)
        per_step.append((time.perf_counter() - t0) * 1e3 / batch)
        step += batch
    fences = mod.get_work_counters()["fence_waits_total"] - c0
    return statistics.median(per_step), fences / (len(per_step) * batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--splats", type=int, nargs="+", default=[5000, 20000])
    ap.add_argument("--pin-after", type=int, default=0, help="rasterize_backward_pin_after (0: keep exploring)")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON here (the extension prints to stdout)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()
    result = {"build": str(sc.ROOT), "reps": args.reps, "pin_after": args.pin_after, "warmup": args.warmup, "cells": {}}
    with tempfile.TemporaryDirectory() as td:
        for n in args.splats:
            ds = sc.build_dataset(Path(td) / f"ds{n}", n_images=6, n_points=n, w=256, h=192)
            for b in args.batches:
                runs = [one_rep(ds, Path(td) / f"o{n}_{b}_{r}", b, args.steps, args.pin_after, args.warmup) for r in range(args.reps)]
                ms = [r[0] for r in runs]
                result["cells"][f"{n}_splats_batch{b}"] = {
                    "ms_per_step_median": round(statistics.median(ms), 3),
                    "ms_per_step_all": [round(x, 3) for x in ms],
                    "fence_waits_per_step": round(statistics.median(r[1] for r in runs), 3),
                }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
