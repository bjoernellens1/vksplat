"""Micro-benchmark of the SSIM term of the train step: ms/step per variant, and PSNR after a fixed step budget.

    python sceneweave/bench_ssim.py --baseline-dir <vksplat dir of 4eda56f> [--reps 7]       # ms/step table
    python sceneweave/bench_ssim.py --psnr --baseline-dir <dir> [--scene <colmap dir>]        # PSNR table

Variants: `old` (the --baseline-dir build, forward/backward SSIM through a 12-float map), `fused` (this build, two-stage SSIM,
defaults), `every2` (ssim_every=2) and `down2` (ssim_downsample=2). Per cell: the median over `--reps` fresh
trainers of (a) the wall ms of one train_step and (b) the GPU-timestamp ms of the SSIM dispatches per step
(`ComputeSSIMGradient`, with its stats / gradient dispatches where the build reports them). Variants of one build
run in one process; the baseline runs in a subprocess (`VKSPLAT_PY_DIR`), since two builds cannot share a process.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

VARIANTS = {
    "old": {},
    "fused": {},
    "every2": {"ssim_every": 2},
    "down2": {"ssim_downsample": 2},
}
SSIM_STAGES = ("ComputeSSIMGradient", "_SSIMStats", "_SSIMGrad")


def stage_totals(mod):
    bd = mod.get_timing_breakdown()
    return {k: bd[k][1] for k in SSIM_STAGES if k in bd}


def timing_rep(ds, out, overrides, steps, warmup):
    mod = sc.new_trainer(ds, out, **overrides)
    n_train = int(mod.num_train)
    for i in range(warmup):
        mod.train_step(i % n_train, i)
    t0 = stage_totals(mod)
    wall = []
    for i in range(steps):
        s = warmup + i
        t = time.perf_counter()
        mod.train_step(s % n_train, s)
        wall.append((time.perf_counter() - t) * 1e3)
    t1 = stage_totals(mod)
    gpu = {k: (t1[k] - t0[k]) * 1e3 / steps for k in t1}
    return statistics.median(wall), gpu


def run_timing(variants, args):
    result = {}
    with tempfile.TemporaryDirectory() as td:
        for w, h in args.sizes:
            for n in args.splats:
                ds = sc.build_dataset(Path(td) / f"ds_{w}_{n}", n_images=6, n_points=n, w=w, h=h)
                for v in variants:
                    runs = [timing_rep(ds, Path(td) / f"o_{w}_{n}_{v}_{r}", VARIANTS[v], args.steps, args.warmup)
                            for r in range(args.reps)]
                    cell = {"wall_ms_per_step": round(statistics.median(r[0] for r in runs), 3)}
                    for k in SSIM_STAGES:
                        vals = [r[1][k] for r in runs if k in r[1]]
                        if vals:
                            cell[f"gpu_ms_{k}"] = round(statistics.median(vals), 3)
                    cell["wall_all"] = [round(r[0], 3) for r in runs]
                    result[f"{w}x{h}_{n}splats_{v}"] = cell
    return result


def psnr_rep(scene, out, overrides, steps):
    from dataclasses import asdict
    sys.path.insert(0, str(sc.ROOT))
    import vksplat
    from simple_trainer import TrainerConfig, join_dir

    mod = vksplat.VkSplat()
    mod.initialize(join_dir(str(sc.ROOT), "shader"), -1)
    cfg = TrainerConfig(
        dataset_dir=str(scene), image_dir=join_dir(str(scene), "images_4"), mask_dir="",
        sparse_dir=join_dir(str(scene), "sparse/0"), output_dir=str(out),
        train_steps=steps, max_steps=steps, enable_viewer=False, save_train_renders=False,
    )
    cfg.refine_start_iter = 100000
    cfg.refine_stop_iter = 200000
    tc = asdict(cfg)
    tc.update(overrides)
    mod.set_train_config(tc)
    n_train = int(mod.num_train)
    t = time.perf_counter()
    for s in range(steps):
        mod.train_step(s % n_train, s)
    mod.render_train(0)  # drain the queue
    elapsed = time.perf_counter() - t
    psnrs = []
    for i in range(int(mod.num_val)):
        mod.render_val(i)
        h, w = np.asarray(mod.get_val_image(i)).shape[:2]
        pred = np.asarray(mod.pixel_state, dtype=np.float32).reshape(h, w, 4)[..., :3]
        ref = np.asarray(mod.get_val_image(i), dtype=np.float32)[..., :3] / 255.0
        psnrs.append(-10 * np.log10(max(float(np.mean((pred - ref) ** 2)), 1e-12)))
    return float(np.mean(psnrs)), elapsed / steps * 1e3, len(mod.opacities)


def run_psnr(variants, args):
    result = {}
    with tempfile.TemporaryDirectory() as td:
        for v in variants:
            runs = [psnr_rep(args.scene, Path(td) / f"p_{v}_{r}", VARIANTS[v], args.psnr_steps)
                    for r in range(args.psnr_reps)]
            result[f"psnr_{v}"] = {
                "psnr_db_median": round(statistics.median(r[0] for r in runs), 3),
                "psnr_db_all": [round(r[0], 3) for r in runs],
                "wall_ms_per_step": round(statistics.median(r[1] for r in runs), 3),
                "num_splats": runs[0][2],
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--splats", type=int, nargs="+", default=[5000, 20000])
    ap.add_argument("--sizes", nargs="+", default=["1200x680", "584x384"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--baseline-dir", type=Path, default=None, help="vksplat dir of the 4eda56f build (variant `old`)")
    ap.add_argument("--psnr", action="store_true", help="PSNR after --psnr-steps on --scene instead of timing")
    ap.add_argument("--scene", type=Path, default=Path("/mnt/cps_persistent1_shared/datasets/public/mipnerf360_v2_dataset/bonsai"))
    ap.add_argument("--psnr-steps", type=int, default=1500)
    ap.add_argument("--psnr-reps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--worker", action="store_true", help="internal: run the given variants and print JSON")
    args = ap.parse_args()
    args.sizes = [tuple(int(x) for x in s.split("x")) for s in args.sizes]

    run = run_psnr if args.psnr else run_timing
    if args.worker:
        print("RESULT_JSON " + json.dumps(run(args.variants, args)))
        return
    result = {"build": str(sc.ROOT), "reps": args.reps}
    local = [v for v in args.variants if v != "old"]
    if local:
        result.update(run(local, args))
    if "old" in args.variants:
        if args.baseline_dir is None:
            sys.exit("variant `old` needs --baseline-dir")
        cmd = [sys.executable, __file__, "--worker", "--variants", "old", "--reps", str(args.reps), "--steps", str(args.steps),
               "--warmup", str(args.warmup), "--splats", *map(str, args.splats), "--sizes", *[f"{w}x{h}" for w, h in args.sizes],
               "--psnr-steps", str(args.psnr_steps), "--psnr-reps", str(args.psnr_reps), "--scene", str(args.scene)]
        if args.psnr:
            cmd.append("--psnr")
        proc = subprocess.run(cmd, env={**os.environ, "VKSPLAT_PY_DIR": str(args.baseline_dir)}, capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON ")]
        if not lines:
            sys.exit(f"baseline worker failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        result.update(json.loads(lines[-1][len("RESULT_JSON "):]))
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
