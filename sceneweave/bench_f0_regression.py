"""F=0 / depth-off training-path regression check: baseline build vs the frozen-prefix + depth build.

Runs the SAME fixed-map workload (no densification, 100 warmup + 300 timed steps) in fresh processes, alternating
builds base,new,base,new,... to cancel thermal/clock drift, and reports per-stage GPU ms/step (Vulkan timestamps),
wall ms/step and the relative difference new vs base. Gate: <= 2% on each stage. Also times render_val with depth on
vs off on the new build.

Driver (from the new worktree):
  python sceneweave/bench_f0_regression.py --base-root /path/to/baseline --new-root /path/to/new \
      --dataset DS --ply PLY --python PIXI_PYTHON [--maps 1,8] [--reps 3] [--pin 1,0]
--pin 1 pins the per-splat raster backward on both builds (fair); --pin 0 uses the scheduler default.
Both roots need the pin_raster_backward_per_splat knob (the baseline gets it as a minimal scheduling-only patch).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

STAGES = ("ProjectionForward", "GenerateKeys", "ComputeTileRanges", "CalculateIndexBufferOffset", "SortRTS",
          "RasterizeForward", "ComputeSSIMGradient", "RasterizeBackward", "FusedProjectionBackwardOptimizerStep",
          "DefaultPostBackward", "CopyTrainImageToDevice")
GATED = ("FusedProjectionBackwardOptimizerStep", "RasterizeForward", "RasterizeBackward", "step_gpu", "wall")
WARMUP, STEPS, RENDERS, ROUNDS = 100, 300, 100, 3


def worker(a):
    sys.path.insert(0, str(Path(a.root) / "vksplat"))
    import vksplat
    from dataclasses import asdict
    from simple_trainer import TrainerConfig, join_dir

    def stage_seconds(m):  # blocks until all prior GPU work is done
        return {k: float(dt) for k, (n, dt) in m.get_timing_breakdown().items()}

    ds = str(Path(a.dataset).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainerConfig(dataset_dir=ds, image_dir=join_dir(ds, "images"), sparse_dir=join_dir(ds, "sparse/0"),
                            mask_dir="", output_dir=tmp, output_ply=str(Path(tmp) / "o.ply"),
                            init_gaussians_ply=a.ply, refine_start_iter=10**9, shuffle_seed=0,
                            train_steps=WARMUP + STEPS, pin_raster_backward_per_splat=bool(a.pin))
        m = vksplat.VkSplat()
        m.initialize(join_dir(str(Path(a.root) / "vksplat"), "shader"), -1)
        m.set_train_config(asdict(cfg))
        for s in range(1, WARMUP + 1):
            m.train_step(s % m.num_train, s)
        before = stage_seconds(m)
        t0 = perf_counter()
        for s in range(WARMUP + 1, WARMUP + STEPS + 1):
            m.train_step(s % m.num_train, s)
        after = stage_seconds(m)
        wall = (perf_counter() - t0) / STEPS * 1e3
        res = {k: 1e3 * (after[k] - before.get(k, 0.0)) / STEPS for k in after if k in STAGES}
        res["step_gpu"], res["wall"] = sum(res.values()), wall
        res["impl"] = m.get_raster_backward_impl()
        # render_val timing: alternate depth off/on blocks (the baseline only has the depth-off signature)
        has_depth = hasattr(vksplat, "SH_REORDER_SIZE")
        modes = (False, True) if has_depth else (False,)
        rend = {x: [] for x in modes}
        for _ in range(ROUNDS):
            for mode in modes:
                b, t0 = stage_seconds(m), perf_counter()
                for i in range(RENDERS):
                    m.render_val(i % m.num_val, mode) if has_depth else m.render_val(i % m.num_val)
                a2 = stage_seconds(m)
                rend[mode].append((1e3 * (a2["RasterizeForward"] - b["RasterizeForward"]) / RENDERS,
                                   (perf_counter() - t0) / RENDERS * 1e3))
        res["render_val"] = {("on" if k else "off"): [statistics.mean(x[j] for x in v) for j in (0, 1)]
                             for k, v in rend.items()}
        m.cleanup()
    print("RESULT " + json.dumps(res))


def run_worker(a, root, ply, pin):
    env = dict(os.environ, VK_SPLAT_FORCE_RADV="1", VK_SPLAT_PYTHON=a.python)
    cmd = ["bash", str(Path(a.new_root) / "sceneweave" / "run_amd.sh"), str(Path(__file__).resolve()), "--worker",
           "--root", root, "--dataset", a.dataset, "--ply", ply, "--pin", str(pin)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise RuntimeError(f"worker failed ({root}):\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return json.loads(line[0][7:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--root")
    p.add_argument("--base-root")
    p.add_argument("--new-root")
    p.add_argument("--dataset", required=True)
    p.add_argument("--ply", help="driver: source PLY (replicated per --maps); worker: PLY to load")
    p.add_argument("--python", help="python with the build deps (pixi env)")
    p.add_argument("--maps", default="1,8", help="replication factors (1 = original, 8 = ~1.2M tiled map)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--pin", default="1,0", help="1 = pin per-splat raster backward on both builds, 0 = scheduler")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.worker:
        a.pin = int(a.pin)
        return worker(a)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_freeze_prefix import replicate_ply
    allres = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k in (int(x) for x in a.maps.split(",")):
            ply = a.ply
            if k > 1:
                ply = str(Path(tmp) / f"map{k}.ply")
                replicate_ply(a.ply, ply, k)
            for pin in (int(x) for x in a.pin.split(",")):
                runs = {"base": [], "new": []}
                for _ in range(a.reps):
                    for tag, root in (("base", a.base_root), ("new", a.new_root)):
                        runs[tag].append(run_worker(a, root, ply, pin))
                allres[f"x{k}_pin{pin}"] = runs
                print(f"\n=== map x{k} ({150000 * k} splats), pin_per_splat={pin}, impl base/new = "
                      f"{runs['base'][0]['impl']} / {runs['new'][0]['impl']} ===")
                print(f"{'stage':40s} {'base ms':>9s} {'new ms':>9s} {'rel diff':>9s} {'median':>8s}  base/new per-run")
                for st in STAGES + ("step_gpu", "wall"):
                    b, n = [r[st] for r in runs["base"]], [r[st] for r in runs["new"]]
                    if not any(b):
                        continue
                    rel = statistics.mean(n) / statistics.mean(b) - 1
                    rel_med = statistics.median(n) / statistics.median(b) - 1
                    flag = ("PASS" if abs(rel) <= 0.02 else "FAIL") if st in GATED else ""
                    print(f"{st:40s} {statistics.mean(b):9.3f} {statistics.mean(n):9.3f} {rel:+9.2%} {rel_med:+8.2%} {flag:4s} "
                          f"{[round(x, 3) for x in b]} / {[round(x, 3) for x in n]}")
                rv = {t: {m_: [r["render_val"][m_][0] for r in runs[t] if m_ in r["render_val"]]
                          for m_ in ("off", "on")} for t in runs}
                off_b, off_n, on_n = (statistics.mean(rv["base"]["off"]), statistics.mean(rv["new"]["off"]),
                                      statistics.mean(rv["new"]["on"]))
                print(f"render_val RasterizeForward GPU ms: base off {off_b:.3f} | new off {off_n:.3f} "
                      f"({off_n / off_b - 1:+.2%} vs base) | new depth on {on_n:.3f} ({on_n / off_n - 1:+.2%} vs new off)")
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
