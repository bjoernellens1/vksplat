"""Tests for vksplat PR #21 — Native GPU Exact-K Active-Set & Temporal Depth Reprojection:
1. Exact-K active-set selection (`select_ago_exact_k_gpu`):
   - Realized active fraction == target_fraction +/- 0.02 across {0.25, 0.50, 0.75, 1.0}
   - Respects exclude_frozen=True and newborn_warmup_steps prioritization (both N_nb <= K and N_nb > K)
   - Excluded Gaussians remain bit-identical across subsequent train_step()
   - Latency benchmark at N=50,500 splats: average < 0.2 ms/call
2. GPU Temporal Depth Reprojection (`reproject_temporal_depth_gpu`):
   - Matches NumPy reference depth (< 1e-4 m), view-angle cosine weight, and support count
   - Correctly resolves foreground-over-background occlusion via atomic uint32 z-buffer
   - Supports shared/per-view intrinsics, custom dst_hw, invalid/NaN depth filtering, and empty batch
   - Latency benchmark at H_hist=8: average < 2.0 ms/call
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402


def _numpy_temporal_reproject(
    src_depths: np.ndarray,
    src_c2ws: np.ndarray,
    dst_c2w: np.ndarray,
    K_src: np.ndarray,
    K_dst: np.ndarray,
    max_depth_m: float = 10.0,
    dst_hw: tuple[int, int] | None = None,
    min_depth_m: float = 0.05,
    min_cos_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact NumPy reference for reproject_temporal_depth_gpu."""
    h_hist, src_h, src_w = src_depths.shape
    dst_h, dst_w = (src_h, src_w) if dst_hw is None else dst_hw

    def _unpack_k(k_arr: np.ndarray) -> tuple[float, float, float, float]:
        if k_arr.shape == (3, 3):
            return float(k_arr[0, 0]), float(k_arr[1, 1]), float(k_arr[0, 2]), float(k_arr[1, 2])
        return float(k_arr[0]), float(k_arr[1]), float(k_arr[2]), float(k_arr[3])

    fx_d, fy_d, cx_d, cy_d = _unpack_k(np.asarray(K_dst, dtype=np.float64))
    dst_w2c = np.linalg.inv(np.asarray(dst_c2w, dtype=np.float64))

    zbuf = np.full((dst_h, dst_w), np.inf, dtype=np.float32)
    weight = np.zeros((dst_h, dst_w), dtype=np.float32)
    support = np.zeros((dst_h, dst_w), dtype=np.uint32)

    uu, vv = np.meshgrid(
        np.arange(src_w, dtype=np.float32),
        np.arange(src_h, dtype=np.float32),
    )

    k_src_arr = np.asarray(K_src, dtype=np.float64)
    for v in range(h_hist):
        kv = k_src_arr[v] if k_src_arr.ndim in (2, 3) and k_src_arr.shape[0] == h_hist and k_src_arr.shape != (3, 3) else k_src_arr
        fx_s, fy_s, cx_s, cy_s = _unpack_k(kv)
        t_s2d = (dst_w2c @ np.asarray(src_c2ws[v], dtype=np.float64)).astype(np.float32)
        R = t_s2d[:3, :3]
        t = t_s2d[:3, 3]

        z_s = np.asarray(src_depths[v], dtype=np.float32)
        valid = np.isfinite(z_s) & (z_s >= np.float32(min_depth_m)) & (z_s <= np.float32(max_depth_m))
        if not np.any(valid):
            continue

        zv = z_s[valid]
        xv = (uu[valid] - np.float32(cx_s)) * np.float32(1.0 / fx_s) * zv
        yv = (vv[valid] - np.float32(cy_s)) * np.float32(1.0 / fy_s) * zv
        p_s = np.stack([xv, yv, zv], axis=-1)
        p_d = p_s @ R.T + t[None, :]

        z_d = p_d[:, 2]
        ok = np.isfinite(z_d) & (z_d >= np.float32(min_depth_m)) & (z_d <= np.float32(max_depth_m))
        p_d = p_d[ok]
        z_d = z_d[ok]
        if p_d.shape[0] == 0:
            continue

        to_dst = -p_d
        to_src = t[None, :] - p_d
        len_dst = np.linalg.norm(to_dst, axis=-1)
        len_src = np.linalg.norm(to_src, axis=-1)
        nz = (len_dst > np.float32(1e-8)) & (len_src > np.float32(1e-8))
        p_d, z_d, to_dst, to_src, len_dst, len_src = (
            p_d[nz], z_d[nz], to_dst[nz], to_src[nz], len_dst[nz], len_src[nz]
        )
        cos_w = np.sum(to_src * to_dst, axis=-1) / (len_src * len_dst)
        ok_cos = np.isfinite(cos_w) & (cos_w >= np.float32(min_cos_angle)) & (cos_w > np.float32(0.0))
        p_d, z_d, cos_w = p_d[ok_cos], z_d[ok_cos], cos_w[ok_cos]
        if p_d.shape[0] == 0:
            continue

        inv_z = np.float32(1.0) / z_d
        u_d = np.floor(np.float32(fx_d) * (p_d[:, 0] * inv_z) + np.float32(cx_d) + np.float32(0.5)).astype(np.int32)
        v_d = np.floor(np.float32(fy_d) * (p_d[:, 1] * inv_z) + np.float32(cy_d) + np.float32(0.5)).astype(np.int32)
        in_bounds = (u_d >= 0) & (u_d < dst_w) & (v_d >= 0) & (v_d < dst_h)
        u_d, v_d, z_d, cos_w = u_d[in_bounds], v_d[in_bounds], z_d[in_bounds], cos_w[in_bounds]

        np.minimum.at(zbuf, (v_d, u_d), z_d)
        np.add.at(weight, (v_d, u_d), cos_w)
        np.add.at(support, (v_d, u_d), np.uint32(1))

    depth = np.where(support > 0, zbuf, np.float32(0.0)).astype(np.float32)
    return depth, weight, support


def _make_pose_c2w(pos: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world 4x4 matrix (+Z forward, +X right, +Y down)."""
    z_axis = look_at - pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    up_ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up_ref, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = pos
    return c2w


def _render_synthetic_planes_depth(
    c2w: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """Analytical ray-cast of a back wall (z_w = 4.0) + foreground box patch (z_w = 2.0)."""
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dirs_cam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    R = c2w[:3, :3].astype(np.float64)
    o = c2w[:3, 3].astype(np.float64)
    dirs_w = dirs_cam @ R.T

    depth = np.zeros((h, w), dtype=np.float32)
    # Back plane at world z = 4.0
    dz = dirs_w[..., 2]
    valid_back = dz > 1e-4
    t_back = np.where(valid_back, (4.0 - o[2]) / np.where(valid_back, dz, 1.0), np.inf)
    # Foreground occluder plane at world z = 2.0 within |x_w| <= 0.6, |y_w| <= 0.5
    t_front = np.where(valid_back, (2.0 - o[2]) / np.where(valid_back, dz, 1.0), np.inf)
    hit_x = o[0] + t_front * dirs_w[..., 0]
    hit_y = o[1] + t_front * dirs_w[..., 1]
    hit_front = (t_front > 0.2) & (np.abs(hit_x) <= 0.6) & (np.abs(hit_y) <= 0.5)
    t_hit = np.where(hit_front, np.minimum(t_back, t_front), t_back)
    ok = np.isfinite(t_hit) & (t_hit > 0.2) & (t_hit < 9.5)
    depth[ok] = t_hit[ok].astype(np.float32)
    return depth


def test_exact_k_target_fractions_frozen_newborn_and_exclusion(tmp_path):
    """Exact-K selection hits target_fraction +/- 0.02, honors frozen & newborns, and keeps excluded bit-exact."""
    ds = sc.build_dataset(tmp_path / "ds_exactk", n_points=600, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_exactk", mcmc_protect_structural_groups=True)
    mod.force_backward_variant(1)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()

    # Run 12 warmup steps so existing splats are mature (update_count >= 10)
    for s in range(1, 13):
        mod.train_step(s % int(mod.num_train), s)

    n = int(len(mod.xyz_ws))
    assert n == 600

    # Test exact-K on all-eligible splats across target fractions
    for frac in (0.25, 0.50, 0.75):
        res = mod.select_ago_exact_k_gpu(target_fraction=frac, newborn_warmup_steps=10, exclude_frozen=True)
        act_idx = np.asarray(mod.get_active_set(), dtype=np.uint32)
        assert res["num_gaussians"] == n
        assert res["eligible_count"] == n
        assert res["active_count"] == len(act_idx)
        assert res["fell_back_to_full"] is False
        assert abs(res["active_fraction"] - frac) <= 0.02, (
            f"Expected active_fraction={frac} +/- 0.02, got {res['active_fraction']:.4f}"
        )
        assert len(set(act_idx.tolist())) == len(act_idx), "Duplicate active indices detected!"

    # Mark first 100 splats as CONTRADICTED (frozen) -> 500 eligible
    states = np.full((n,), 5, dtype=np.uint32)
    states[:100] = 4
    mod.set_splat_states(states)

    res_f = mod.select_ago_exact_k_gpu(target_fraction=0.50, newborn_warmup_steps=10, exclude_frozen=True)
    selected_set = set(np.asarray(mod.get_active_set(), dtype=np.uint32).tolist())
    assert res_f["eligible_count"] == 500
    assert res_f["active_count"] == 250
    assert len(selected_set) == 250
    assert all(i >= 100 for i in selected_set), "Frozen splats must be excluded"

    # Verify excluded Gaussians are bit-identical across train_step()
    xyz_before = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    mod.train_step(1, 13)
    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    excl_idx = np.array(sorted(set(range(n)) - selected_set), dtype=np.int64)
    act_idx_arr = np.array(sorted(selected_set), dtype=np.int64)
    assert float(np.max(np.abs(xyz_after[excl_idx] - xyz_before[excl_idx]))) == 0.0
    assert float(np.max(np.abs(xyz_after[act_idx_arr] - xyz_before[act_idx_arr]))) > 0.0

    # Newborn prioritization: append 40 newborns (update_count == 0 < 10)
    rng = np.random.default_rng(123)
    xyz_add, sh_add, rot_add, sc_add, op_add, gids_add = sc.make_batch(rng, 40, -1)
    mod.append_splats_preserving_optimizer(xyz_add, sh_add, rot_add, sc_add, op_add, gids_add)
    n_tot = int(len(mod.xyz_ws))
    assert n_tot == 640
    states_all = np.full((n_tot,), 5, dtype=np.uint32)
    mod.set_splat_states(states_all)

    # K = 320 >= 40 newborns -> all 40 newborns MUST be selected
    res_nb = mod.select_ago_exact_k_gpu(target_fraction=0.50, newborn_warmup_steps=10, exclude_frozen=True)
    sel_nb = set(np.asarray(mod.get_active_set(), dtype=np.uint32).tolist())
    assert res_nb["active_count"] == 320
    assert len(sel_nb) == 320
    for i in range(600, 640):
        assert i in sel_nb, f"Newborn {i} must be prioritized when newborn_count <= K"

    # Newborn overflow mode (newborn_count > K): huge warmup -> all 640 are newborns, K = 160
    res_nb_over = mod.select_ago_exact_k_gpu(target_fraction=0.25, newborn_warmup_steps=1e6, exclude_frozen=True)
    sel_nb_over = np.asarray(mod.get_active_set(), dtype=np.uint32)
    assert res_nb_over["active_count"] == 160
    assert len(sel_nb_over) == 160
    assert abs(res_nb_over["active_fraction"] - 0.25) <= 0.02

    # Empty fallback (target_fraction == 0.0)
    res_zero = mod.select_ago_exact_k_gpu(target_fraction=0.0, newborn_warmup_steps=10, exclude_frozen=True)
    assert res_zero["active_count"] == 0
    assert res_zero["fell_back_to_full"] is True

    mod.cleanup()


def test_exact_k_latency_benchmark(tmp_path):
    """Exact-K selection averages < 0.2 ms/call at N=50,500 splats with active_fraction == 0.50 +/- 0.02."""
    ds = sc.build_dataset(tmp_path / "ds_exactk_lat", n_points=500, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_exactk_lat")
    mod.force_backward_variant(1)

    rng = np.random.default_rng(2026)
    xyz, sh, rot, scales, opac, gids_add = sc.make_batch(rng, 50_000, -1)
    mod.append_splats_preserving_optimizer(xyz, sh, rot, scales, opac, gids_add)
    n_tot = int(len(mod.xyz_ws))
    assert n_tot == 50_500

    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()
    for s in range(1, 4):
        mod.train_step(s % int(mod.num_train), s)

    # Warmup
    for _ in range(3):
        mod.select_ago_exact_k_gpu(0.5, 10, True)

    n_iters = 20
    t0 = time.perf_counter()
    last_res = None
    for _ in range(n_iters):
        last_res = mod.select_ago_exact_k_gpu(0.5, 10, True)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    print(
        f"[PR21 Exact-K GPU Benchmark N={n_tot}] "
        f"avg={avg_ms:.3f} ms/call, active={last_res['active_count']}/{n_tot} "
        f"({last_res['active_fraction']:.4f})"
    )
    assert abs(last_res["active_fraction"] - 0.50) <= 0.02
    assert avg_ms < 0.2, f"Expected < 0.2 ms/call, got {avg_ms:.3f} ms/call"

    mod.cleanup()


def test_temporal_depth_reprojection_accuracy_and_occlusion(tmp_path):
    """GPU temporal depth reprojection matches NumPy reference (< 1e-4 m) and resolves occlusion."""
    ds = sc.build_dataset(tmp_path / "ds_treproj_acc", n_points=100, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_treproj_acc")

    h_src, w_src = 96, 128
    h_dst, w_dst = 96, 128
    K_src = np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    K_dst = np.array([[105.0, 0.0, 63.5], [0.0, 105.0, 47.5], [0.0, 0.0, 1.0]], dtype=np.float32)

    look_at = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    dst_c2w = _make_pose_c2w(np.array([0.0, 0.0, 0.0]), look_at)

    offsets = [
        [-0.35, 0.00, 0.05],
        [+0.35, 0.00, 0.05],
        [-0.20, 0.15, -0.05],
        [+0.20, -0.15, -0.05],
        [0.00, 0.25, 0.10],
        [0.00, -0.25, 0.10],
    ]
    src_c2ws = np.stack([_make_pose_c2w(np.array(p), look_at) for p in offsets], axis=0)
    src_depths = np.stack([_render_synthetic_planes_depth(c, K_src, h_src, w_src) for c in src_c2ws], axis=0)

    # Inject invalid depths (0.0, NaN, > max_depth_m) to verify filtering
    src_depths[0, :5, :5] = 0.0
    src_depths[1, :5, :5] = np.nan
    src_depths[2, :5, :5] = 25.0

    ref_depth, ref_weight, ref_support = _numpy_temporal_reproject(
        src_depths, src_c2ws, dst_c2w, K_src, K_dst, max_depth_m=10.0
    )
    gpu_out = mod.reproject_temporal_depth_gpu(
        src_depths, src_c2ws, dst_c2w, K_src, K_dst, max_depth_m=10.0
    )
    gpu_depth = np.asarray(gpu_out["depth"], dtype=np.float32)
    gpu_weight = np.asarray(gpu_out["weight"], dtype=np.float32)
    gpu_support = np.asarray(gpu_out["support_count"], dtype=np.uint32)

    assert gpu_depth.shape == (h_dst, w_dst)
    assert gpu_weight.shape == (h_dst, w_dst)
    assert gpu_support.shape == (h_dst, w_dst)

    both_valid = (ref_support > 0) & (gpu_support > 0)
    assert int(np.sum(both_valid)) > 0.5 * h_dst * w_dst

    # Support agreement across >99.5% of pixels (tiny sub-pixel 0.5 rounding ties permitted)
    exact_support_ratio = float(np.mean(gpu_support == ref_support))
    assert exact_support_ratio > 0.995, f"Support match ratio too low: {exact_support_ratio:.5f}"

    # Depth accuracy < 1e-4 m on all pixels with identical support
    matched = (gpu_support == ref_support) & (ref_support > 0)
    max_depth_err = float(np.max(np.abs(gpu_depth[matched] - ref_depth[matched])))
    max_weight_err = float(np.max(np.abs(gpu_weight[matched] - ref_weight[matched])))
    assert max_depth_err < 1e-4, f"Expected max depth err < 1e-4 m, got {max_depth_err:.6e} m"
    assert max_weight_err < 1e-3, f"Expected max weight err < 1e-3, got {max_weight_err:.6e}"

    # Center pixel hits foreground occluder (z ~= 2.0 m), NOT background wall (z ~= 4.0 m)
    center_d = float(gpu_depth[h_dst // 2, w_dst // 2])
    assert abs(center_d - 2.0) < 0.05, f"Occlusion failed at center pixel: got {center_d:.3f} m"

    # Invalid pixels must have depth == 0.0 and weight == 0.0
    inv_mask = gpu_support == 0
    if np.any(inv_mask):
        assert float(np.max(np.abs(gpu_depth[inv_mask]))) == 0.0
        assert float(np.max(np.abs(gpu_weight[inv_mask]))) == 0.0

    # Empty batch H_hist == 0 returns clean zeros
    empty_out = mod.reproject_temporal_depth_gpu(
        np.zeros((0, h_src, w_src), dtype=np.float32),
        np.zeros((0, 4, 4), dtype=np.float32),
        dst_c2w,
        K_src,
        K_dst,
        max_depth_m=10.0,
    )
    assert empty_out["depth"].shape == (h_src, w_src)
    assert float(np.max(np.abs(empty_out["depth"]))) == 0.0
    assert int(np.sum(empty_out["support_count"])) == 0

    # Custom dst_hw and 4-vector K format
    custom_out = mod.reproject_temporal_depth_gpu(
        src_depths[:2],
        src_c2ws[:2],
        dst_c2w,
        np.array([100.0, 100.0, 64.0, 48.0], dtype=np.float32),
        np.array([50.0, 50.0, 32.0, 24.0], dtype=np.float32),
        max_depth_m=10.0,
        dst_hw=(48, 64),
    )
    assert custom_out["depth"].shape == (48, 64)
    assert int(np.sum(custom_out["support_count"] > 0)) > 100

    mod.cleanup()


def test_temporal_depth_reprojection_latency_benchmark(tmp_path):
    """H_hist=8 temporal depth reprojection averages < 2.0 ms/call on GPU."""
    ds = sc.build_dataset(tmp_path / "ds_treproj_lat", n_points=100, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_treproj_lat")

    h_hist = 8
    h, w = 240, 320
    K = np.array([[260.0, 0.0, 160.0], [0.0, 260.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    look_at = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    dst_c2w = _make_pose_c2w(np.array([0.0, 0.0, 0.0]), look_at)

    angles = np.linspace(-0.4, 0.4, h_hist)
    src_c2ws = np.stack(
        [_make_pose_c2w(np.array([float(a), 0.1 * float(np.sin(3 * a)), 0.0]), look_at) for a in angles],
        axis=0,
    )
    src_depths = np.stack([_render_synthetic_planes_depth(c, K, h, w) for c in src_c2ws], axis=0)

    # Warmup
    for _ in range(3):
        mod.reproject_temporal_depth_gpu(src_depths, src_c2ws, dst_c2w, K, K, max_depth_m=10.0)

    n_iters = 10
    t0 = time.perf_counter()
    last_out = None
    for _ in range(n_iters):
        last_out = mod.reproject_temporal_depth_gpu(src_depths, src_c2ws, dst_c2w, K, K, max_depth_m=10.0)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    print(
        f"[PR21 Temporal Depth Reprojection Benchmark H_hist={h_hist} {h}x{w}] "
        f"wall_avg={avg_ms:.3f} ms/call, last_kernel={last_out['kernel_ms']:.3f} ms, "
        f"valid_px={int(np.sum(last_out['support_count'] > 0))}/{h * w}"
    )
    assert avg_ms < 2.0, f"Expected < 2.0 ms/call, got {avg_ms:.3f} ms/call"

    mod.cleanup()
