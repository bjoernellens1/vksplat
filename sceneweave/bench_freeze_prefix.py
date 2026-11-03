"""Frozen-prefix benchmark on a FIXED Gaussian map and fixed camera workload (no densification).

Usage (from vksplat-db0/vksplat, Vulkan env via run_amd.sh):
  VK_SPLAT_FORCE_RADV=1 bash ../sceneweave/run_amd.sh ../sceneweave/bench_freeze_prefix.py \
      --dataset DS --init-ply PLY [--steps 300] [--fractions 0,0.25,0.5,0.75,0.9] [--replicate 8] [--refine-start 500]

Per frozen fraction (prefix rounded down to a multiple of SH_REORDER_SIZE) it reports GPU time per stage (Vulkan
timestamps, ms/step), wall ms/step, the raster-backward implementation used, forward RGB/depth identity of the
initial map vs F=0, and the equivalence of the active splats' first-step update vs F=0. The first row is F=0 with the
scheduler's default raster backward, the second row is F=0 with the per-splat variant pinned (the implementation
every F>0 run uses); speedups are quoted against the pinned row. The script asserts fractions[0]==0, bitwise
RGB/depth identity of the initial forward and active-update equivalence, so a failed expectation aborts the run. --replicate K tiles the PLY K times with small position jitter to
emulate a final-size map (the oracle's 150k surfels are ~11% of a 1.3M map).
"""
import argparse
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

VK = Path(__file__).resolve().parents[1] / "vksplat"
sys.path.insert(0, str(VK))
import vksplat  # noqa: E402
from ckpt_trace import depth_from_render  # noqa: E402
from simple_trainer import TrainerConfig, join_dir  # noqa: E402

TOP_LEVEL = ("ProjectionForward", "GenerateKeys", "ComputeTileRanges", "CalculateIndexBufferOffset", "SortRTS",
             "RasterizeForward", "ComputeSSIMGradient", "RasterizeBackward",
             "FusedProjectionBackwardOptimizerStep", "DefaultPostBackward", "CopyTrainImageToDevice")
PARAMS = ("xyz_ws", "scales", "rotations", "opacities", "sh_coeffs")


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


def replicate_ply(src, dst, k, jitter=0.01, seed=0):
    raw = Path(src).read_bytes()
    head, _, body = raw.partition(b"end_header\n")
    names = [l.split()[-1] for l in head.decode().splitlines() if l.startswith("property")]
    data = np.frombuffer(body, dtype="<f4").reshape(-1, len(names))
    rng = np.random.default_rng(seed)
    out = [data]
    for _ in range(k - 1):
        d = data.copy()
        d[:, :3] += rng.normal(0, jitter, (len(d), 3)).astype("<f4")
        out.append(d)
    out = np.concatenate(out)
    h = head.decode().replace(f"element vertex {len(data)}", f"element vertex {len(out)}")
    Path(dst).write_bytes(h.encode() + b"end_header\n" + out.astype("<f4").tobytes())
    return len(out)


def stage_seconds(m):
    return {k: float(dt) for k, (n, dt) in m.get_timing_breakdown().items()}


def make(args, ply, F, pin, out_dir):
    ds = str(Path(args.dataset).resolve())
    cfg = TrainerConfig(dataset_dir=ds, image_dir=join_dir(ds, args.image_dir), sparse_dir=join_dir(ds, "sparse/0"),
                        mask_dir="", output_dir=out_dir, output_ply=str(Path(out_dir) / "bench.ply"),
                        init_gaussians_ply=ply, freeze_prefix_count=F, pin_raster_backward_per_splat=pin,
                        refine_start_iter=args.refine_start, shuffle_seed=0, train_steps=args.steps,
                        refine_stop_iter=args.refine_stop)
    m = vksplat.VkSplat()
    m.initialize(join_dir(str(VK), "shader"), -1)
    m.set_train_config(asdict(cfg))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--image-dir", default="images")
    p.add_argument("--init-ply", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--fractions", default="0,0.25,0.5,0.75,0.9")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--refine-start", type=int, default=10**9, help="default: no densification (fixed map)")
    p.add_argument("--refine-stop", type=int, default=15000)
    args = p.parse_args()
    fractions = [float(x) for x in args.fractions.split(",")]
    assert fractions[0] == 0, "the first fraction is the F=0 reference and must be 0"
    sh = vksplat.SH_REORDER_SIZE  # prefix granularity

    with tempfile.TemporaryDirectory() as tmp:
        ply = args.init_ply
        if args.replicate > 1:
            ply = str(Path(tmp) / "big.ply")
            n = replicate_ply(args.init_ply, ply, args.replicate)
        else:
            n = int(Path(ply).read_bytes().partition(b"end_header")[0].decode().split("element vertex")[1].split()[0])
        print(f"map: {n} splats, {args.steps} timed steps (+{args.warmup} warmup), refine_start={args.refine_start}, "
              f"prefix granularity {sh}", flush=True)

        # (F, pin_raster_backward_per_splat): scheduled F=0, pinned F=0, then every F>0 (pinned implicitly)
        specs = [(0, False), (0, True)] + [(int(f * n) // sh * sh, True) for f in fractions[1:]]
        base = {}
        rows = []
        for F, pin in specs:
            m = make(args, ply, F, pin, tmp)
            # identical initial forward scene: RGB + depth of val view 0 must match F=0 bit-for-bit
            m.render_val(0, True)
            st, pd = np.array(m.pixel_state), np.array(m.pixel_depth)
            depth, _ = depth_from_render(st, pd, 0.5)
            # first-step update of the active splats vs the pinned F=0 run (same init, same view, same raster backward)
            m.train_step(0, 1)
            s1 = {k: np.array(getattr(m, k)) for k in PARAMS}
            if F == 0 and pin:
                base = dict(st=st, depth=depth, s1=s1)
            if base:
                same_fwd = bool(np.array_equal(bits(st), bits(base["st"]))
                                and np.array_equal(bits(np.nan_to_num(depth)), bits(np.nan_to_num(base["depth"]))))
                assert same_fwd, f"F={F}: initial forward RGB/depth differs from the F=0 run"
                eq = {k: (float(np.abs(base["s1"][k][F:] - s1[k][F:]).max()),
                          int((np.abs(base["s1"][k][F:] - s1[k][F:]) > 1e-7).sum())) for k in PARAMS}
                for k in PARAMS:  # float-atomic summation order moves a few elements by ~1 ulp; rotations (Adam's
                    # first step is +-lr on ~zero gradients) can flip sign, so they get a looser count bound
                    bound = (1e-4 if k == "rotations" else 1e-5) * s1[k][F:].size
                    assert eq[k][1] <= bound and (k == "rotations" or eq[k][0] <= 1e-6), \
                        f"F={F}: active {k} update differs from F=0: {eq[k]}"
            else:
                same_fwd, eq = None, {}
            for s in range(2, args.warmup + 2):
                m.train_step(s % m.num_train, s)
            before = stage_seconds(m)  # blocks until all prior GPU work is done
            t0 = perf_counter()
            for s in range(args.warmup + 2, args.warmup + 2 + args.steps):
                m.train_step(s % m.num_train, s)
            after = stage_seconds(m)
            wall = (perf_counter() - t0) / args.steps * 1e3
            stages = {k: 1e3 * (after[k] - before.get(k, 0.0)) / args.steps for k in after if k in TOP_LEVEL}
            rows.append((F, pin, wall, stages, same_fwd, eq, m.get_raster_backward_impl()))
            m.cleanup()

    cols = [c for c in TOP_LEVEL if any(c in r[3] for r in rows)]
    print("\nGPU ms/step by stage (Vulkan timestamps) and wall ms/step")
    print("F".rjust(9), "frac".rjust(5), "pin".rjust(4), "raster_bwd".rjust(13), "wall".rjust(8), "gpu_sum".rjust(8),
          *[c[:14].rjust(14) for c in cols])
    for F, pin, wall, stages, _, _, impl in rows:
        print(str(F).rjust(9), f"{F / n:5.2f}", str(int(pin)).rjust(4), impl.rjust(13), f"{wall:8.3f}",
              f"{sum(stages.values()):8.3f}", *[f"{stages.get(c, 0.0):14.3f}" for c in cols])
    sched, pinned = rows[0], rows[1]
    fused = "FusedProjectionBackwardOptimizerStep"
    for ref, label in ((pinned, "pinned"), (sched, "scheduled")):
        print(f"\nspeedup vs F=0 {label} ({ref[6]}); ceiling for the fused stage = 1/(1-frac)")
        for F, _, wall, stages, *_ in rows[2:]:
            print(f"F={F}: fused {ref[3][fused] / stages[fused]:.2f}x (ceiling {1 / (1 - F / n):.2f}x), "
                  f"step gpu {sum(ref[3].values()) / sum(stages.values()):.3f}x, wall {ref[2] / wall:.3f}x")
    print("\nF=0 (pinned) stage share of GPU time:",
          ", ".join(f"{c}={100 * v / sum(pinned[3].values()):.1f}%" for c, v in pinned[3].items()))
    print("\nforward identity vs pinned F=0 (initial map RGB+depth bitwise, asserted) and first-step active update "
          "(max abs diff, #elems>1e-7)")
    for F, pin, _, _, same, eq, _ in rows[1:]:
        print(f"F={F}: forward_identical={same}  " + "  ".join(f"{k}=({v[0]:.2g},{v[1]})" for k, v in eq.items()))
    print(f"\ninitial surfels / map size: {n} splats; a 150k oracle prefix is {150000 / 1.3e6:.1%} of a ~1.3M final map")


if __name__ == "__main__":
    main()
