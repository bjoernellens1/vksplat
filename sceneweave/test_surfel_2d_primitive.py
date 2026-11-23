"""Verification tests for V17.15 Track D2: Native SURFEL_2D primitive & 2D->3D promotion in VkSplat."""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

VKSPLAT_DIR = Path(__file__).resolve().parents[1] / "vksplat"


def _make_minimal_colmap_scene(root: Path, n_pts: int = 32) -> Path:
    sparse = root / "sparse" / "0"
    imgs = root / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    imgs.mkdir(parents=True, exist_ok=True)

    # Synthetic 64x64 image with a warm wall gradient
    im = np.zeros((64, 64, 3), dtype=np.uint8)
    im[:, :, 0] = 180
    im[:, :, 1] = 140
    im[:, :, 2] = 100
    cv2.imwrite(str(imgs / "frame_000.png"), im)

    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 64 64 60.0 60.0 32.0 32.0\n", encoding="utf-8"
    )
    (sparse / "images.txt").write_text(
        "# Image list\n1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_000.png\n\n", encoding="utf-8"
    )
    rng = np.random.default_rng(7)
    lines = ["# Points3D"]
    for i in range(1, n_pts + 1):
        x = float(rng.uniform(-0.6, 0.6))
        y = float(rng.uniform(-0.6, 0.6))
        z = float(2.0 + rng.uniform(-0.02, 0.02))
        lines.append(f"{i} {x:.5f} {y:.5f} {z:.5f} 160 130 100 0")
    (sparse / "points3D.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_surfel_2d_zero_normal_scale_and_promotion_to_3d():
    from dataclasses import asdict
    import sys

    if str(VKSPLAT_DIR) not in sys.path:
        sys.path.insert(0, str(VKSPLAT_DIR))
    import vksplat  # type: ignore
    from simple_trainer import TrainerConfig, join_dir  # type: ignore

    with tempfile.TemporaryDirectory(prefix="vksplat_surfel2d_") as td:
        root = _make_minimal_colmap_scene(Path(td) / "scene", n_pts=32)
        out = Path(td) / "out"
        out.mkdir()
        mod = vksplat.VkSplat()
        mod.initialize(join_dir(str(VKSPLAT_DIR), "shader"), -1)
        cfg = TrainerConfig(
            dataset_dir=str(root),
            image_dir=join_dir(str(root), "images"),
            mask_dir="",
            sparse_dir=join_dir(str(root), "sparse/0"),
            output_dir=str(out),
            train_steps=100,
            max_steps=200,
            enable_viewer=False,
            save_train_renders=False,
            init_reference_count=32,
        )
        cfg.refine_start_iter = 99999
        cfg.refine_stop_iter = 99999
        mod.set_train_config(asdict(cfg))

        n = len(np.asarray(mod.xyz_ws).reshape(-1, 3))
        assert n == 32
        # Half SURFEL_2D (rows 0..15), half GAUSSIAN_3D (rows 16..31)
        mask_2d = np.zeros(n, dtype=np.uint8)
        mask_2d[:16] = 1
        mod.set_splat_surfel_2d_mask(mask_2d)

        st0 = np.asarray(mod.get_splat_states(), dtype=np.uint32)
        assert np.all((st0[:16] & 0x00010000) != 0)
        assert np.all((st0[16:] & 0x00010000) == 0)

        sc0 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
        assert np.allclose(sc0[:16, 2], 0.0, atol=1e-7)
        assert np.all(sc0[16:, 2] > 0.0)

        # Train 30 steps: SURFEL_2D must keep scales.z == 0.0 while rendering finite pixels
        # (even across Morton sorting, which permutes splat_state alongside scales_opacs!)
        for step in range(30):
            mod.train_step(0, step)
        mod.render_train(0)
        st1 = np.asarray(mod.get_splat_states(), dtype=np.uint32)
        is_2d_1 = (st1 & 0x00010000) != 0
        assert int(is_2d_1.sum()) == 16
        sc1 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
        assert np.all(np.isfinite(sc1))
        assert np.allclose(sc1[is_2d_1, 2], 0.0, atol=1e-7), "SURFEL_2D normal scale must remain strictly 0.0"
        assert np.all(sc1[~is_2d_1, 2] > 0.0), "GAUSSIAN_3D normal scale must remain volumetric"

        # Promote 8 of the SURFEL_2D rows to GAUSSIAN_3D with sigma_n = 0.025
        rows_2d = np.flatnonzero(is_2d_1).astype(np.uint32)
        rows_prom = rows_2d[:8]
        rows_keep = rows_2d[8:]
        sig_n = np.full(8, 0.025, dtype=np.float32)
        mod.promote_surfels_to_3d(rows_prom, sig_n)

        st2 = np.asarray(mod.get_splat_states(), dtype=np.uint32)
        assert np.all((st2[rows_prom] & 0x00010000) == 0), "Promoted rows must clear SURFEL_2D flag"
        assert np.all((st2[rows_prom] & 0x00020000) != 0), "Promoted rows must set GPU-resident PROMOTED_3D bit 17"
        assert np.all((st2[rows_keep] & 0x00010000) != 0), "Unpromoted planar rows must remain SURFEL_2D"
        assert np.all((st2[rows_keep] & 0x00020000) == 0), "Unpromoted planar rows must not set PROMOTED_3D"
        sc2 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
        assert np.allclose(sc2[rows_prom, 2], 0.025, atol=1e-5)
        assert np.allclose(sc2[rows_keep, 2], 0.0, atol=1e-7)

        # Verify PROMOTED_3D bit 17 survives snapshot/restore and further Morton-sorted training steps!
        snap = mod.snapshot_optimizer_state()
        for step in range(30, 50):
            mod.train_step(0, step)
        mod.render_train(0)
        st3 = np.asarray(mod.get_splat_states(), dtype=np.uint32)
        is_2d_3 = (st3 & 0x00010000) != 0
        is_prom_3 = (st3 & 0x00020000) != 0
        assert int(is_2d_3.sum()) == 8
        assert int(is_prom_3.sum()) == 8
        sc3 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
        assert np.all(np.isfinite(sc3))
        assert np.all(sc3[~is_2d_3, 2] > 0.005)
        assert np.allclose(sc3[is_2d_3, 2], 0.0, atol=1e-7)

        mod.restore_optimizer_state(snap)
        st4 = np.asarray(mod.get_splat_states(), dtype=np.uint32)
        assert int(((st4 & 0x00010000) != 0).sum()) == 8
        assert int(((st4 & 0x00020000) != 0).sum()) == 8
