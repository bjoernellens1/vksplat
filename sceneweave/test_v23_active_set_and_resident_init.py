"""Tests for V23 native Vulkan features:
1. True GPU Active-Set Compaction (compact_mode0_active_set + optimizer dispatch)
2. GPU-Resident Appearance Initialization (append_and_init_surfels_vulkan)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402
from test_v22_vulkan_sh_and_timestamps import _make_lookat_c2w  # noqa: E402


def _make_synthetic_rgbd_views(
    means_world: np.ndarray,
    normals_world: np.ndarray,
    n_views: int = 4,
    h_rgb: int = 64,
    w_rgb: int = 80,
    h_d: int = 48,
    w_d: int = 64,
):
    """Build consistent synthetic multi-view RGB-D frames observing means_world."""
    n_pts = means_world.shape[0]
    poses_c2w = np.zeros((n_views, 4, 4), dtype=np.float32)
    w2cs = np.zeros((n_views, 4, 4), dtype=np.float32)
    cam_os = np.zeros((n_views, 3), dtype=np.float32)
    k_rgbs = np.zeros((n_views, 3, 3), dtype=np.float32)
    k_depths = np.zeros((n_views, 3, 3), dtype=np.float32)
    rgb_views = np.zeros((n_views, h_rgb, w_rgb, 3), dtype=np.uint8)
    depth_views = np.zeros((n_views, h_d, w_d), dtype=np.float32)

    target = np.mean(means_world, axis=0).astype(np.float64)
    for v in range(n_views):
        ang = 2.0 * np.pi * v / max(n_views, 1)
        cam_pos = target + np.array([2.2 * np.cos(ang), 0.35 * np.sin(2.0 * ang), 2.2 * np.sin(ang)], dtype=np.float64)
        c2w = _make_lookat_c2w(cam_pos, target)
        w2c = np.linalg.inv(c2w)
        poses_c2w[v] = c2w.astype(np.float32)
        w2cs[v] = w2c.astype(np.float32)
        cam_os[v] = cam_pos.astype(np.float32)

        fx_r, fy_r = 70.0, 70.0
        cx_r, cy_r = w_rgb * 0.5, h_rgb * 0.5
        k_rgbs[v] = np.array([[fx_r, 0.0, cx_r], [0.0, fy_r, cy_r], [0.0, 0.0, 1.0]], dtype=np.float32)

        fx_d, fy_d = 56.0, 56.0
        cx_d, cy_d = w_d * 0.5, h_d * 0.5
        k_depths[v] = np.array([[fx_d, 0.0, cx_d], [0.0, fy_d, cy_d], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Smooth view-dependent RGB pattern
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h_rgb), np.linspace(0.0, 1.0, w_rgb), indexing="ij")
        r_ch = np.clip(0.2 + 0.5 * xx + 0.1 * v, 0.0, 1.0)
        g_ch = np.clip(0.25 + 0.5 * yy, 0.0, 1.0)
        b_ch = np.clip(0.7 - 0.3 * xx - 0.2 * yy + 0.05 * v, 0.0, 1.0)
        rgb_views[v] = np.rint(np.stack([r_ch, g_ch, b_ch], axis=-1) * 255.0).astype(np.uint8)

        # Paint exact depth at projected points so visibility & depth tolerance pass cleanly
        p_cam = means_world.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3][None, :]
        z = p_cam[:, 2]
        valid = z > 0.25
        u_d = np.rint(fx_d * (p_cam[:, 0] / np.maximum(z, 0.25)) + cx_d).astype(np.int64)
        v_d = np.rint(fy_d * (p_cam[:, 1] / np.maximum(z, 0.25)) + cy_d).astype(np.int64)
        depth_img = np.full((h_d, w_d), float(np.median(z[valid])) if np.any(valid) else 2.2, dtype=np.float32)
        for i in range(n_pts):
            if valid[i] and 0 <= u_d[i] < w_d and 0 <= v_d[i] < h_d:
                depth_img[v_d[i], u_d[i]] = float(z[i])
        depth_views[v] = depth_img

    return rgb_views, depth_views, k_rgbs, k_depths, poses_c2w, w2cs, cam_os


def test_compact_mode0_active_set_75pct_frozen_correctness_and_speedup(tmp_path):
    """75% frozen splats: gpu_active_buffer_bound==True, bit-exact frozen params, reduced optimizer_ms."""
    # 1) Correctness & bit-exact preservation with 75% frozen
    ds = sc.build_dataset(tmp_path / "ds_corr", n_points=400, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out_corr", mcmc_protect_structural_groups=True)
    mod.set_freeze_skip_backward(True)

    n = int(len(mod.xyz_ws))
    assert n == 400
    n_frozen = 300  # 75%
    n_active = 100  # 25%

    gids = np.full((n,), -1, dtype=np.int32)
    gids[:200] = 2  # LOCKED structural group
    mod.set_splat_group_ids(gids)
    mod.set_group_state(2, 2)

    states = np.full((n,), 5, dtype=np.uint32)  # MEASURED_FREE
    states[:200] = 0xFFFFFFFF  # inherits LOCKED group state
    states[200:n_frozen] = 4   # CONTRADICTED -> frozen
    mod.set_splat_states(states)

    stats = mod.compact_mode0_active_set()
    assert stats["total_splats"] == n
    assert stats["frozen_splats"] == n_frozen
    assert stats["active_splats"] == n_active
    assert stats["active_dispatch_splats"] == n_active
    assert abs(stats["active_fraction"] - 0.25) < 1e-9
    assert stats["gpu_active_buffer_bound"] is True

    xyz_before = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_before = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_before = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    for s in range(1, 10):
        mod.train_step(s % int(mod.num_train), s)

    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_after = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_after = np.asarray(mod.rotations, dtype=np.float32)
    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)

    # Frozen splats must be bit-exact untouched
    assert np.array_equal(xyz_after[:n_frozen], xyz_before[:n_frozen])
    assert np.array_equal(sh_after[:n_frozen], sh_before[:n_frozen])
    assert np.array_equal(rot_after[:n_frozen], rot_before[:n_frozen])
    assert np.array_equal(so_after[:n_frozen], so_before[:n_frozen])

    # Active splats must have updated
    assert float(np.max(np.abs(xyz_after[n_frozen:] - xyz_before[n_frozen:]))) > 1e-6
    assert float(np.max(np.abs(sh_after[n_frozen:] - sh_before[n_frozen:]))) > 1e-6

    # 100% frozen: skips optimizer cleanly
    states_all_frozen = np.full((n,), 4, dtype=np.uint32)
    mod.set_splat_states(states_all_frozen)
    stats_zero = mod.compact_mode0_active_set()
    assert stats_zero["active_dispatch_splats"] == 0
    assert stats_zero["gpu_active_buffer_bound"] is True
    xyz_z0 = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    mod.train_step(1, 11)
    assert np.array_equal(np.asarray(mod.xyz_ws, dtype=np.float32), xyz_z0)

    # 2) GPU timestamp speedup benchmark: 0% frozen vs 75% frozen on 120k splats
    ds_bench = sc.build_dataset(tmp_path / "ds_bench", n_points=500, w=64, h=48)
    rng = np.random.default_rng(23)
    n_add = 120_000

    def _run_bench(freeze_frac: float) -> tuple[float, dict]:
        m = sc.new_trainer(ds_bench, tmp_path / f"out_b_{int(freeze_frac*100)}", mcmc_protect_structural_groups=True)
        m.set_freeze_skip_backward(True)
        xyz, sh, rot, scales, opac, gids_add = sc.make_batch(rng, n_add, -1)
        m.append_splats_preserving_optimizer(xyz, sh, rot, scales, opac, gids_add)
        tot = int(len(m.xyz_ws))
        sts = np.full((tot,), 5, dtype=np.uint32)
        n_frz = int(tot * freeze_frac)
        if n_frz > 0:
            sts[:n_frz] = 4
        m.set_splat_states(sts)
        cstats = m.compact_mode0_active_set()
        # Warmup 3 steps
        for s in range(1, 4):
            m.train_step(s % int(m.num_train), s)
        m.reset_gpu_stage_timestamps()
        for s in range(4, 24):
            m.train_step(s % int(m.num_train), s)
        ts = m.get_gpu_stage_timestamps_ms()
        return float(ts["optimizer_ms"]), cstats

    opt_ms_0, cstats_0 = _run_bench(0.0)
    opt_ms_75, cstats_75 = _run_bench(0.75)

    assert cstats_0["gpu_active_buffer_bound"] is False
    assert cstats_75["gpu_active_buffer_bound"] is True
    assert cstats_75["active_dispatch_splats"] == cstats_75["total_splats"] - cstats_75["frozen_splats"]
    print(f"\n[V23 Bench] 120.5k splats x 20 steps: optimizer_ms(0% frozen)={opt_ms_0:.3f} ms, "
          f"optimizer_ms(75% frozen)={opt_ms_75:.3f} ms (speedup={opt_ms_0 / max(opt_ms_75, 1e-6):.2f}x)")
    assert opt_ms_75 < opt_ms_0, f"Expected 75% frozen optimizer_ms ({opt_ms_75:.3f}) < 0% frozen ({opt_ms_0:.3f})"


def test_append_and_init_surfels_vulkan_matches_project_and_fit_sh(tmp_path):
    """append_and_init_surfels_vulkan matches project_and_fit_sh_vulkan (mae < 1e-5) and appends GPU-resident."""
    # Use n_points=37 (non-multiple of SH_REORDER_SIZE=32) to stress partial reorder block boundaries
    ds = sc.build_dataset(tmp_path / "ds_init", n_points=37, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out_init", mcmc_protect_structural_groups=True)

    # Run a few steps so existing splats have non-zero Adam moments
    for s in range(1, 5):
        mod.train_step(s % int(mod.num_train), s)

    old_n = int(len(mod.xyz_ws))
    assert old_n == 37
    xyz_old = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_old = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_old = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_old = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    rng = np.random.default_rng(123)
    m = 135  # spans across multiple SH_REORDER_SIZE=32 blocks starting from offset 37
    means_world = rng.uniform([-0.4, -0.3, -0.3], [0.4, 0.3, 0.3], size=(m, 3)).astype(np.float32)
    normals_world = np.zeros((m, 3), dtype=np.float32)
    normals_world[:, 2] = 1.0  # facing +Z
    # Mix some outward normals so views hit varied cosines
    normals_world[m // 2 :, 0] = 0.7
    normals_world[m // 2 :, 2] = 0.7
    normals_world /= np.linalg.norm(normals_world, axis=1, keepdims=True)

    scales_norm = rng.uniform(0.01, 0.04, size=(m, 3)).astype(np.float32)
    quats_norm = rng.normal(size=(m, 4)).astype(np.float32)
    quats_norm /= np.linalg.norm(quats_norm, axis=1, keepdims=True)
    opacities = rng.uniform(0.3, 0.85, size=(m,)).astype(np.float32)
    group_ids = np.full((m,), 3, dtype=np.int32)
    splat_states = np.full((m,), 2, dtype=np.uint32)
    fallback_rgb = rng.uniform(0.1, 0.9, size=(m, 3)).astype(np.float32)

    rgb_views, depth_views, k_rgbs, k_depths, poses_c2w, w2cs, cam_os = _make_synthetic_rgbd_views(
        means_world, normals_world, n_views=4
    )

    # Custom affine dp_transform
    dp_transform = np.array(
        [
            [0.5, 0.0, 0.0, 0.1],
            [0.0, 0.5, 0.0, -0.05],
            [0.0, 0.0, 0.5, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    sh_reg = 1e-2
    depth_tol = 0.12

    # Reference: existing project_and_fit_sh_vulkan
    ref = mod.project_and_fit_sh_vulkan(
        means_world,
        normals_world,
        rgb_views,
        depth_views,
        k_rgbs,
        k_depths,
        w2cs,
        cam_os,
        fallback_rgb,
        sh_degree=1,
        reg_lambda=sh_reg,
        depth_tol_m=depth_tol,
    )
    ref_sh = np.asarray(ref["sh_coeffs"], dtype=np.float32)
    ref_valid = int(np.sum(np.asarray(ref["obs_count"]) > 0))
    assert ref_valid > 0

    mod.reset_gpu_stage_timestamps()
    res = mod.append_and_init_surfels_vulkan(
        means_world,
        normals_world,
        scales_norm,
        quats_norm,
        opacities,
        group_ids,
        splat_states,
        fallback_rgb,
        rgb_views,
        depth_views,
        k_rgbs,
        k_depths,
        poses_c2w,
        dp_transform,
        sh_reg=sh_reg,
        depth_tol=depth_tol,
    )

    assert res["n_appended"] == m
    assert res["gpu_resident"] is True
    assert res["n_valid_sh"] == ref_valid
    assert res["kernel_ms"] >= 0.0
    assert res["total_ms"] >= res["kernel_ms"]
    assert abs(res["mean_wls_rmse"] - float(ref["wls_sh_fit_rmse"])) < 1e-6

    ts = mod.get_gpu_stage_timestamps_ms()
    assert ts["appear_init_ms"] > 0.0

    # Verify total splat count & bit-exact preservation of existing splats
    assert int(len(mod.xyz_ws)) == old_n + m
    xyz_all = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_all = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_all = np.asarray(mod.rotations, dtype=np.float32)
    so_all = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)

    assert np.array_equal(xyz_all[:old_n], xyz_old)
    assert np.array_equal(sh_all[:old_n], sh_old)
    assert np.array_equal(rot_all[:old_n], rot_old)
    assert np.array_equal(so_all[:old_n], so_old)

    # Verify appended geometry & SH match reference
    expected_xyz_norm = (means_world @ dp_transform[:3, :3].T + dp_transform[:3, 3][None, :]).astype(np.float32)
    assert np.max(np.abs(xyz_all[old_n:] - expected_xyz_norm)) < 1e-6
    assert np.array_equal(rot_all[old_n:], quats_norm)
    assert np.allclose(so_all[old_n:, :3], scales_norm, atol=1e-6)
    assert np.allclose(so_all[old_n:, 3], opacities, atol=1e-6)

    sh_mae = float(np.mean(np.abs(sh_all[old_n:] - ref_sh)))
    sh_max_err = float(np.max(np.abs(sh_all[old_n:] - ref_sh)))
    print(f"\n[V23 Resident Init] n_appended={m}, n_valid_sh={res['n_valid_sh']}, "
          f"sh_mae={sh_mae:.2e}, sh_max_err={sh_max_err:.2e}, "
          f"kernel_ms={res['kernel_ms']:.3f}, total_ms={res['total_ms']:.3f}, "
          f"appear_init_ms={ts['appear_init_ms']:.3f}")
    assert sh_mae < 1e-5, f"SH MAE {sh_mae} >= 1e-5"
    assert sh_max_err < 1e-5, f"SH max err {sh_max_err} >= 1e-5"

    # Verify subsequent training steps run cleanly on the GPU-resident appended surfels
    for s in range(5, 8):
        mod.train_step(s % int(mod.num_train), s)
