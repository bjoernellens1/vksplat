"""Micro-benchmark: keyframe upload (replace_train_frame), blocking vs asynchronous.

    python sceneweave/bench_frame_upload.py [--reps 7] [--out result.json]

Per frame size one trainer holds 20k splats and 4 keyframe slots. Per arm and repetition: `--iters` timed
`replace_train_frame` calls (upload ms), and `--iters` timed "replace + train_steps(4)" pairs (end-to-end ms).
Reported: median over repetitions of the per-repetition median.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import append_scenario as sc


def one_size(ds, out, w, h, reps, iters):
    mod = sc.new_trainer(ds, out)
    mod.append_splats_preserving_optimizer(*sc.make_batch(np.random.default_rng(7), 20000, 11))
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8) for _ in range(4)]
    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = -2.0
    n = int(mod.num_train)

    def replace(k):
        mod.replace_train_frame(k % n, frames[k % 4], c2w, 0.8 * w, 0.8 * h, w / 2, h / 2)

    result = {}
    for arm, flag in (("sync", False), ("async", True)):
        mod.set_async_frame_upload(flag)
        for k in range(8):  # warm-up: staging slots, device buffers, pipelines
            replace(k)
            mod.train_steps([k % n] * 4, 0)
        upload, e2e = [], []
        waits0 = mod.get_work_counters()["frame_upload_waits"]
        for _ in range(reps):
            ts = []
            for k in range(iters):
                t0 = time.perf_counter()
                replace(k)
                ts.append((time.perf_counter() - t0) * 1e3)
                mod.train_steps([0] * 4, 0)  # drain so that each upload starts from an idle queue
            upload.append(statistics.median(ts))
            ts = []
            for k in range(iters):
                t0 = time.perf_counter()
                replace(k)
                mod.train_steps([k % n] * 4, 4 * k)
                ts.append((time.perf_counter() - t0) * 1e3)
            e2e.append(statistics.median(ts))
        result[arm] = {"upload_ms": round(statistics.median(upload), 3), "upload_ms_all": [round(x, 3) for x in upload],
                       "upload_plus_train_steps4_ms": round(statistics.median(e2e), 3),
                       "e2e_ms_all": [round(x, 3) for x in e2e],
                       "frame_upload_waits": mod.get_work_counters()["frame_upload_waits"] - waits0}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = {"build": str(sc.ROOT), "reps": args.reps, "iters": args.iters}
    with tempfile.TemporaryDirectory() as td:
        for w, h in ((1200, 680), (584, 384)):
            ds = sc.build_dataset(Path(td) / f"ds{w}", n_images=6, n_points=100, w=w, h=h, structured=True)
            result[f"{w}x{h}"] = one_size(ds, Path(td) / f"o{w}", w, h, args.reps, args.iters)
            print(f"{w}x{h}", json.dumps({a: {k: v for k, v in r.items() if not k.endswith("_all")}
                                         for a, r in result[f"{w}x{h}"].items()}), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
