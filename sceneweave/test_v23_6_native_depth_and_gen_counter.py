"""Tests for V23.6 P0 native Vulkan engine features:
1. Active-Set Generation Counter & Mutation Safety (map_generation / active_set_generation)
2. Native Vulkan Gaussian Depth Rendering (render_val_depth & render_pose_rgbd)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402


def test_active_set_generation_counter_churn_regression(tmp_path):
    """Verify generation counter prevents stale active-index dispatch across append/prune/reorder churn."""
    ds = sc.build_dataset(tmp_path / "ds_churn", n_points=400, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out_churn", mcmc_protect_structural_groups=True)
    mod.set_freeze_skip_backward(True)

    n0 = int(len(mod.xyz_ws))
    assert n0 == 400
    n_frozen = 200  # 50% frozen

    # Freeze first 200 splats: 100 via LOCKED group 2, 100 via CONTRADICTED state 4
    gids = np.full((n0,), -1, dtype=np.int32)
    gids[:100] = 2
    gids[200:250] = 7  # will be retracted & compacted later
    mod.set_splat_group_ids(gids)
    mod.set_group_state(2, 2)  # LOCKED

    states = np.full((n0,), 5, dtype=np.uint32)  # MEASURED_FREE
    states[:100] = 0xFFFFFFFF  # inherits LOCKED group state
    states[100:n_frozen] = 4  # CONTRADICTED -> frozen
    mod.set_splat_states(states)

    stats1 = mod.compact_mode0_active_set()
    assert stats1["total_splats"] == n0
    assert stats1["frozen_splats"] == n_frozen
    assert stats1["active_splats"] == n0 - n_frozen
    assert stats1["active_dispatch_splats"] == n0 - n_frozen
    assert stats1["gpu_active_buffer_bound"] is True
    assert stats1["active_set_generation"] == stats1["map_generation"]
    assert mod.active_set_generation == mod.map_generation == stats1["map_generation"]
    gen_after_compact1 = int(stats1["map_generation"])

    # Append 100 new trainable splats placed inside the existing scene cloud so they project to cameras
    rng = np.random.default_rng(236)
    xyz_existing = np.asarray(mod.xyz_ws, dtype=np.float32)
    n_add = 100
    xyz_add = xyz_existing[:n_add] + rng.normal(0.0, 0.02, size=(n_add, 3)).astype(np.float32)
    sh_add = rng.normal(0.0, 0.15, size=(n_add, 16, 3)).astype(np.float32)
    rot_add = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_add, 1))
    sc_add = np.full((n_add, 3), 0.04, dtype=np.float32)
    op_add = np.full((n_add,), 0.6, dtype=np.float32)
    gids_add = np.full((n_add,), -1, dtype=np.int32)
    states_add = np.full((n_add,), 5, dtype=np.uint32)  # MEASURED_FREE

    mod.append_splats_preserving_optimizer(
        xyz_add, sh_add, rot_add, sc_add, op_add, gids_add, splat_states=states_add
    )
    assert int(len(mod.xyz_ws)) == n0 + n_add
    gen_after_append = int(mod.map_generation)
    assert gen_after_append > gen_after_compact1
    assert int(mod.active_set_generation) == gen_after_compact1  # stale!

    # Train WITHOUT re-compacting: stale active indices must NOT be dispatched,
    # so newly appended splats [400:500] MUST receive non-zero optimizer updates!
    xyz_before = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_before = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_before = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    for s in range(1, 9):
        mod.train_step(s % int(mod.num_train), s)

    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_after = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_after = np.asarray(mod.rotations, dtype=np.float32)
    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)

    new_xyz_delta = float(np.max(np.abs(xyz_after[n0:] - xyz_before[n0:])))
    new_sh_delta = float(np.max(np.abs(sh_after[n0:] - sh_before[n0:])))
    assert new_xyz_delta > 1e-6, f"Newly appended splats got zero xyz update ({new_xyz_delta}); stale active set?"
    assert new_sh_delta > 1e-6, f"Newly appended splats got zero SH update ({new_sh_delta}); stale active set?"
    # Frozen splats must remain bit-exact untouched
    assert np.array_equal(xyz_after[:n_frozen], xyz_before[:n_frozen])
    assert np.array_equal(sh_after[:n_frozen], sh_before[:n_frozen])
    assert np.array_equal(rot_after[:n_frozen], rot_before[:n_frozen])
    assert np.array_equal(so_after[:n_frozen], so_before[:n_frozen])

    # Prune/reorder mutation: retract group 7 (50 splats) and compact_retracted()
    # compact_retracted removes both RETRACTED group 7 (50 splats) and CONTRADICTED state 4 (100 splats)
    mod.retract_group_sparse(7)
    gen_after_retract = int(mod.map_generation)
    assert gen_after_retract > gen_after_append
    mod.compact_retracted()
    gen_after_prune = int(mod.map_generation)
    assert gen_after_prune > gen_after_retract
    n_expected_after_prune = n0 + n_add - 50 - 100
    assert int(len(mod.xyz_ws)) == n_expected_after_prune

    # Re-compact active set and verify generation matches again
    # Remaining frozen splats are the 100 LOCKED group-2 splats at indices [:100]
    n_frozen_after_prune = 100
    stats2 = mod.compact_mode0_active_set()
    assert stats2["map_generation"] == gen_after_prune
    assert stats2["active_set_generation"] == stats2["map_generation"]
    assert stats2["gpu_active_buffer_bound"] is True
    assert stats2["frozen_splats"] == n_frozen_after_prune
    assert stats2["active_splats"] == n_expected_after_prune - n_frozen_after_prune

    xyz_b2 = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_b2 = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_b2 = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_b2 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    for s in range(9, 17):
        mod.train_step(s % int(mod.num_train), s)

    xyz_a2 = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_a2 = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_a2 = np.asarray(mod.rotations, dtype=np.float32)
    so_a2 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)

    # Frozen params bit-exact, trainable params updated
    assert np.array_equal(xyz_a2[:n_frozen_after_prune], xyz_b2[:n_frozen_after_prune])
    assert np.array_equal(sh_a2[:n_frozen_after_prune], sh_b2[:n_frozen_after_prune])
    assert np.array_equal(rot_a2[:n_frozen_after_prune], rot_b2[:n_frozen_after_prune])
    assert np.array_equal(so_a2[:n_frozen_after_prune], so_b2[:n_frozen_after_prune])
    assert float(np.max(np.abs(xyz_a2[n_frozen_after_prune:] - xyz_b2[n_frozen_after_prune:]))) > 1e-6
    assert float(np.max(np.abs(sh_a2[n_frozen_after_prune:] - sh_b2[n_frozen_after_prune:]))) > 1e-6


def _world_to_norm_points(dp_row_major: np.ndarray, xyz_world: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply VkSplat dataparser_transform (row-major 4x4) to world points."""
    dp = np.asarray(dp_row_major, dtype=np.float64)
    scale = float(np.linalg.norm(dp[:3, 0]))
    xyz_norm = (xyz_world.astype(np.float64) @ dp[:3, :3].T + dp[:3, 3][None, :]).astype(np.float32)
    return xyz_norm, scale


def _make_plane_gaussians(
    z_world: float,
    x_half: float,
    y_half: float,
    grid_n: int,
    scale_xy_m: float,
    scale_z_m: float,
    opacity: float,
    rgb_val: tuple[float, float, float],
    dp_col_major: np.ndarray,
):
    """Build a dense grid of flat Gaussians on z = z_world facing the origin camera."""
    xs = np.linspace(-x_half, x_half, grid_n, dtype=np.float32)
    ys = np.linspace(-y_half, y_half, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xyz_w = np.stack([xx.ravel(), yy.ravel(), np.full(grid_n * grid_n, z_world, dtype=np.float32)], axis=-1)
    xyz_n, dp_scale = _world_to_norm_points(dp_col_major, xyz_w)
    m = xyz_n.shape[0]
    sh = np.zeros((m, 16, 3), dtype=np.float32)
    c0 = 0.28209479177387814
    sh[:, 0, 0] = (rgb_val[0] - 0.5) / c0
    sh[:, 0, 1] = (rgb_val[1] - 0.5) / c0
    sh[:, 0, 2] = (rgb_val[2] - 0.5) / c0
    rot = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (m, 1))
    scales = np.tile(
        np.array([[scale_xy_m * dp_scale, scale_xy_m * dp_scale, scale_z_m * dp_scale]], dtype=np.float32),
        (m, 1),
    )
    opac = np.full((m, 1), opacity, dtype=np.float32)
    return xyz_n, sh, rot, scales, opac


def test_native_depth_analytic_plane_and_occlusion(tmp_path):
    """Verify render_pose_rgbd and render_val_depth accuracy on an analytic Z=2.0m plane and a 2-layer occlusion."""
    from dataclasses import asdict

    sys.path.insert(0, str(sc.ROOT))
    import vksplat
    from simple_trainer import TrainerConfig, join_dir

    ds = sc.build_dataset(tmp_path / "ds_depth", n_points=100, w=64, h=48)
    out_dir = tmp_path / "out_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = vksplat.VkSplat()
    mod.initialize(join_dir(str(sc.ROOT), "shader"), -1)
    cfg = TrainerConfig(
        dataset_dir=str(ds),
        image_dir=join_dir(str(ds), "images"),
        mask_dir="",
        sparse_dir=join_dir(str(ds), "sparse/0"),
        output_dir=str(out_dir),
        train_steps=50,
        max_steps=500,
        enable_viewer=False,
        save_train_renders=False,
    )
    train_meta = mod.set_train_config(asdict(cfg))
    dp_col = np.asarray(train_meta["dataparser_transform"], dtype=np.float32)
    w, h = 64, 48
    fx, fy = 60.0, 60.0
    cx, cy = w * 0.5, h * 0.5
    c2w = np.eye(4, dtype=np.float32)

    # 1) Single flat plane at Z = 2.0 m covering the full viewport (x in [-1.4, 1.4], y in [-1.1, 1.1])
    xyz_p, sh_p, rot_p, sc_p, op_p = _make_plane_gaussians(
        z_world=2.0,
        x_half=1.4,
        y_half=1.1,
        grid_n=28,
        scale_xy_m=0.10,
        scale_z_m=0.05,
        opacity=0.95,
        rgb_val=(0.8, 0.3, 0.2),
        dp_col_major=dp_col,
    )
    mod.set_gauss_params(xyz_p, rot_p, sc_p, op_p, sh_p)

    # Register the exact same camera as val frame 0
    mod.clear_val_frames()
    dummy_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    val_idx = mod.append_val_frame(dummy_rgba, c2w, fx, fy, cx, cy, "plane_val.png")
    assert val_idx == 0

    out_pose = mod.render_pose_rgbd(c2w, fx, fy, cx, cy, w, h)
    out_val = mod.render_val_depth(0)

    for label, out in [("render_pose_rgbd", out_pose), ("render_val_depth", out_val)]:
        assert out["depth_eval_source"] == "native_rendered"
        d_exp = np.asarray(out["depth_expected"], dtype=np.float32)
        d_first = np.asarray(out["depth_first_surface"], dtype=np.float32)
        alpha = np.asarray(out["alpha"], dtype=np.float32)
        rgb = np.asarray(out["rgb"], dtype=np.float32)
        assert d_exp.shape == (h, w)
        assert d_first.shape == (h, w)
        assert alpha.shape == (h, w)
        assert rgb.shape == (h, w, 3)

        # >80% of plane pixels should have alpha > 0.9
        opaque_mask = alpha > 0.9
        opaque_frac = float(np.mean(opaque_mask))
        assert opaque_frac > 0.80, f"{label}: opaque_frac={opaque_frac:.3f} <= 0.80"

        err_exp = float(np.max(np.abs(d_exp[opaque_mask] - 2.0)))
        err_first = float(np.max(np.abs(d_first[opaque_mask] - 2.0)))
        assert err_exp < 0.02, f"{label}: max |depth_expected - 2.0| = {err_exp:.5f} >= 0.02"
        assert err_first < 0.02, f"{label}: max |depth_first_surface - 2.0| = {err_first:.5f} >= 0.02"

    # render_pose_rgbd and render_val_depth must agree identically for the same camera
    assert np.allclose(out_pose["depth_expected"], out_val["depth_expected"], atol=1e-5)
    assert np.allclose(out_pose["depth_first_surface"], out_val["depth_first_surface"], atol=1e-5)
    assert np.allclose(out_pose["alpha"], out_val["alpha"], atol=1e-5)
    assert np.allclose(out_pose["rgb"], out_val["rgb"], atol=1e-5)

    # 2) Two-layer occlusion scene:
    #    - Semi-transparent foreground layer at Z = 1.5 m (single Gaussian per cell -> alpha ~ 0.65 > 0.5)
    #    - Opaque background layer at Z = 3.5 m (alpha ~ 0.98)
    #    At the center pixel, depth_first_surface must snap to foreground (1.5 m),
    #    while depth_expected blends foreground (1.5 m) and background (3.5 m) -> strictly > 1.8 m and < 3.2 m!
    xyz_fg, sh_fg, rot_fg, sc_fg, op_fg = _make_plane_gaussians(
        z_world=1.5,
        x_half=0.0,
        y_half=0.0,
        grid_n=1,
        scale_xy_m=0.60,
        scale_z_m=0.60,
        opacity=0.65,
        rgb_val=(0.1, 0.9, 0.1),
        dp_col_major=dp_col,
    )
    xyz_bg, sh_bg, rot_bg, sc_bg, op_bg = _make_plane_gaussians(
        z_world=3.5,
        x_half=0.0,
        y_half=0.0,
        grid_n=1,
        scale_xy_m=1.40,
        scale_z_m=1.40,
        opacity=0.98,
        rgb_val=(0.1, 0.1, 0.9),
        dp_col_major=dp_col,
    )
    mod.set_gauss_params(
        np.concatenate([xyz_fg, xyz_bg], axis=0),
        np.concatenate([rot_fg, rot_bg], axis=0),
        np.concatenate([sc_fg, sc_bg], axis=0),
        np.concatenate([op_fg, op_bg], axis=0),
        np.concatenate([sh_fg, sh_bg], axis=0),
    )

    occ = mod.render_pose_rgbd(c2w, fx, fy, cx, cy, w, h)
    # Inspect the central 4x4 patch directly behind the origin ray
    cy_i, cx_i = h // 2, w // 2
    d_first_c = float(np.median(occ["depth_first_surface"][cy_i - 2 : cy_i + 2, cx_i - 2 : cx_i + 2]))
    d_exp_c = float(np.median(occ["depth_expected"][cy_i - 2 : cy_i + 2, cx_i - 2 : cx_i + 2]))
    alpha_c = float(np.median(occ["alpha"][cy_i - 2 : cy_i + 2, cx_i - 2 : cx_i + 2]))

    assert alpha_c > 0.95, f"Center occlusion alpha too low: {alpha_c:.4f}"
    # Analytic values at center:
    #   fg weight = 0.65, bg weight = (1 - 0.65) * 0.98 = 0.343, sum = 0.993
    #   d_first = 1.50 m
    #   d_expected = (0.65 * 1.5 + 0.343 * 3.5) / 0.993 = 2.1908 m
    assert abs(d_first_c - 1.5) < 0.02, f"Expected first-surface snap to 1.5m, got {d_first_c:.4f}m"
    assert d_exp_c > d_first_c + 0.4, f"Expected depth_expected ({d_exp_c:.4f}m) to blend toward background (>1.9m)"
    assert abs(d_exp_c - 2.19) < 0.05, f"Expected analytic blended depth ~2.19m, got {d_exp_c:.4f}m"
