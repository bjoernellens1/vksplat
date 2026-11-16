"""Shared scenario for the reserved-capacity append tests and benchmark (synthetic COLMAP scene)."""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# VKSPLAT_PY_DIR selects another vksplat build (e.g. the fd1a50d baseline) for A/B runs
ROOT = Path(os.environ.get("VKSPLAT_PY_DIR") or Path(__file__).resolve().parents[1] / "vksplat")


def build_dataset(root: Path, n_images: int = 4, n_points: int = 2000, w: int = 64, h: int = 48) -> Path:
    import cv2

    img_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    f, cx, cy = 0.8 * w, w / 2, h / 2
    (sparse_dir / "cameras.txt").write_text(f"# Camera list\n1 PINHOLE {w} {h} {f} {f} {cx} {cy}\n")
    rng = np.random.default_rng(11)
    lines = ["# Image list"]
    for i in range(n_images):
        name = f"frame_{i:04d}.png"
        cv2.imwrite(str(img_dir / name), rng.integers(40, 210, size=(h, w, 3), dtype=np.uint8))
        lines += [f"{i + 1} 1.0 0.0 0.0 0.0 {0.12 * i:.4f} 0.0 2.0 1 {name}", ""]
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")
    pts = ["# 3D point list"]
    for pid in range(1, n_points + 1):
        p = rng.uniform([-0.8, -0.6, -0.5], [0.8, 0.6, 0.5])
        c = rng.integers(60, 220, size=3)
        pts.append(f"{pid} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0.01 1 0")
    (sparse_dir / "points3D.txt").write_text("\n".join(pts) + "\n")
    return root


def new_trainer(root: Path, out: Path, pin_after: int = 0, **overrides):
    sys.path.insert(0, str(ROOT))
    import vksplat
    from simple_trainer import TrainerConfig, join_dir

    mod = vksplat.VkSplat()
    mod.initialize(join_dir(str(ROOT), "shader"), -1)
    cfg = TrainerConfig(
        dataset_dir=str(root), image_dir=join_dir(str(root), "images"), mask_dir="",
        sparse_dir=join_dir(str(root), "sparse/0"), output_dir=str(out),
        train_steps=50, max_steps=500, enable_viewer=False, save_train_renders=False,
    )
    if pin_after:
        cfg.rasterize_backward_pin_after = pin_after  # ignored by builds that predate the field
    cfg.refine_start_iter = 100000
    cfg.refine_stop_iter = 200000
    train_config = asdict(cfg)
    train_config.update(overrides)  # also keys that only the native config knows (mcmc_protect_structural_groups)
    mod.set_train_config(train_config)
    return mod


def make_batch(rng: np.random.Generator, m: int, gid: int):
    xyz = rng.uniform([-0.8, -0.6, -0.5], [0.8, 0.6, 0.5], size=(m, 3)).astype(np.float32)
    sh = rng.normal(0.0, 0.2, size=(m, 16, 3)).astype(np.float32)
    rot = rng.normal(size=(m, 4)).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    sc = rng.uniform(0.01, 0.05, size=(m, 3)).astype(np.float32)
    op = rng.uniform(0.2, 0.8, size=m).astype(np.float32)
    return xyz, sh, rot, sc, op, np.full(m, gid, dtype=np.int32)


def snapshot(mod) -> dict:
    return {
        "xyz": np.asarray(mod.xyz_ws, dtype=np.float32).copy(),
        "rot": np.asarray(mod.rotations, dtype=np.float32).copy(),
        "scales": np.asarray(mod.scales, dtype=np.float32).copy(),
        "opac": np.asarray(mod.opacities, dtype=np.float32).copy(),
        "sh": np.asarray(mod.sh_coeffs, dtype=np.float32).copy(),
        "gids": np.asarray(mod.get_splat_group_ids(), dtype=np.int32).copy(),
    }


def run(root: Path, out: Path, batches: int, m: int, steps_between: int, final_steps: int,
        seed: int = 5, merge: bool = False) -> dict:
    """Append `batches` x `m` splats (one group id per batch), training `steps_between` steps after each
    append and `final_steps` at the end. merge=True appends all batches in one call (needs steps_between=0)."""
    mod = new_trainer(root, out)
    n_train = int(mod.num_train)
    rng = np.random.default_rng(seed)
    parts = [make_batch(rng, m, 10 + b) for b in range(batches)]
    step = 0
    if merge:
        assert steps_between == 0
        mod.append_splats_preserving_optimizer(*[np.concatenate([p[i] for p in parts]) for i in range(6)])
    else:
        for part in parts:
            mod.append_splats_preserving_optimizer(*part)
            for _ in range(steps_between):
                mod.train_step(step % n_train, step)
                step += 1
    for _ in range(final_steps):
        mod.train_step(step % n_train, step)
        step += 1
    return snapshot(mod)


if __name__ == "__main__":
    import tempfile

    dest = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        ds = build_dataset(Path(td) / "ds", n_points=2000)
        snap = run(ds, Path(td) / "out", batches=6, m=37, steps_between=2, final_steps=6)
    np.savez(dest, **snap)
