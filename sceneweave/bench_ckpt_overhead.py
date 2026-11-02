"""Per-step wall time of the training loop: baseline trainer (git rev) vs. current trainer with checkpoint_steps=None.

Usage (via run_amd.sh so the Vulkan env matches training):
  bash sceneweave/run_amd.sh ../sceneweave/bench_ckpt_overhead.py --dataset DS [--image-dir images] [--steps 300] [--reps 5]
Runs alternate base/new in fresh subprocesses; reports median per-step ms and the relative overhead (target <= 1%).
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

VK = Path(__file__).resolve().parents[1] / "vksplat"


def child(args):
    sys.path.insert(0, args.trainer_dir)
    import simple_trainer as st
    cfg = st.TrainerConfig(
        dataset_dir=args.dataset, image_dir=args.image_dir, sparse_dir="sparse/0", output_dir=args.out,
        train_steps=args.steps, max_steps=30000, enable_viewer=False, save_train_renders=False)
    cfg.image_dir = st.join_dir(cfg.dataset_dir, cfg.image_dir)
    cfg.sparse_dir = st.join_dir(cfg.dataset_dir, cfg.sparse_dir)
    cfg.mask_dir = ""
    cfg.output_ply = str(Path(args.out) / "splat.ply")
    cfg.shuffle_seed = 0
    st.train(cfg)
    print("BENCH_JSON", json.dumps(json.load(open(Path(args.out) / "train.json"))["time_elapsed"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--image-dir", default="images")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--base-rev", default="48d80d0")
    p.add_argument("--child", action="store_true")
    p.add_argument("--trainer-dir")
    p.add_argument("--out")
    args = p.parse_args()
    if args.child:
        return child(args)

    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp) / "base"
        base_dir.mkdir()
        (base_dir / "simple_trainer.py").write_text(subprocess.run(
            ["git", "-C", str(VK), "show", f"{args.base_rev}:vksplat/simple_trainer.py"],
            check=True, capture_output=True, text=True).stdout)
        (base_dir / "shader").symlink_to(VK / "shader")  # train() resolves shaders next to simple_trainer.py
        for so in VK.glob("vksplat*.so"):
            (base_dir / so.name).symlink_to(so)
        times = {"base": [], "new": []}
        for rep in range(args.reps):
            for variant, tdir in (("base", base_dir), ("new", VK)):
                out = Path(tmp) / f"{variant}{rep}"
                r = subprocess.run(
                    [sys.executable, __file__, "--child", "--dataset", args.dataset, "--image-dir", args.image_dir,
                     "--steps", str(args.steps), "--trainer-dir", str(tdir), "--out", str(out)],
                    cwd=tdir, capture_output=True, text=True)
                line = [l for l in r.stdout.splitlines() if l.startswith("BENCH_JSON")]
                if not line:
                    sys.exit(f"{variant} run failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                times[variant].append(1000 * float(line[0].split()[1]) / args.steps)
                print(f"rep {rep} {variant}: {times[variant][-1]:.4f} ms/step", flush=True)
    mb, mn = statistics.median(times["base"]), statistics.median(times["new"])
    print(f"median ms/step  base={mb:.4f}  new(checkpoint_steps=None)={mn:.4f}  overhead={100 * (mn - mb) / mb:+.2f}%")


if __name__ == "__main__":
    main()
