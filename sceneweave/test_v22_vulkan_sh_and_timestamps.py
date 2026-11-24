"""Tests for native Slang/Vulkan SH appearance init, GPU stage timestamps, and Mode-0 active-set hook."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def _make_lookat_c2w(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos
    return c2w


def _cpu_reference_project_and_fit_sh(
    means_world: np.ndarray,
    normals_world: np.ndarray | None,
    rgb_views_u8: np.ndarray,
    depth_views_f32: np.ndarray,
    k_rgb_f32: np.ndarray,
    k_depth_f32: np.ndarray,
    w2c_f32: np.ndarray,
    cam_o_f32: np.ndarray,
    fallback_rgb_f32: np.ndarray,
    sh_degree: int = 1,
    reg_lambda: float = 0.25,
    depth_tol_m: float = 0.12,
    min_cos_angle: float = 0.15,
) -> dict:
    """Exact CPU reference matching sceneweave.gpu_appearance_init.project_and_fit_sh_gpu."""
    n_pts = means_world.shape[0]
    n_views, h_rgb, w_rgb, _ = rgb_views_u8.shape
    _, h_d, w_d = depth_views_f32.shape
    n_basis = (sh_degree + 1) ** 2

    unit_n = None
    has_n = None
    if normals_world is not None:
        n_lens = np.linalg.norm(normals_world.astype(np.float64), axis=1)
        has_n = n_lens > 1e-8
        unit_n = np.where(
            has_n[:, None],
            normals_world.astype(np.float64) / np.maximum(n_lens, 1e-8)[:, None],
            0.0,
        )

    atwa = np.zeros((n_pts, n_basis, n_basis), dtype=np.float64)
    atwy = np.zeros((n_pts, n_basis, 3), dtype=np.float64)
    rgb_wsum = np.zeros((n_pts, 3), dtype=np.float64)
    weight_sum = np.zeros((n_pts,), dtype=np.float64)
    obs_count = np.zeros((n_pts,), dtype=np.int32)
    records = []

    xyz = means_world.astype(np.float64)
    for v in range(n_views):
        r = w2c_f32[v, :3, :3].astype(np.float64)
        t = w2c_f32[v, :3, 3].astype(np.float64)
        p_cam = xyz @ r.T + t[None, :]
        z = p_cam[:, 2]
        valid = np.isfinite(z) & (z > 0.2)
        inv_z = np.where(valid, 1.0 / np.maximum(z, 0.2), 0.0)
        xn = p_cam[:, 0] * inv_z
        yn = p_cam[:, 1] * inv_z

        u_d = k_depth_f32[v, 0, 0] * xn + k_depth_f32[v, 0, 2]
        v_d = k_depth_f32[v, 1, 1] * yn + k_depth_f32[v, 1, 2]
        u_di = np.rint(u_d).astype(np.int64)
        v_di = np.rint(v_d).astype(np.int64)

        u_rgb = k_rgb_f32[v, 0, 0] * xn + k_rgb_f32[v, 0, 2]
        v_rgb = k_rgb_f32[v, 1, 1] * yn + k_rgb_f32[v, 1, 2]

        in_b = (
            (u_di >= 0) & (u_di < w_d) & (v_di >= 0) & (v_di < h_d)
            & (u_rgb >= 0.0) & (u_rgb <= w_rgb - 1) & (v_rgb >= 0.0) & (v_rgb <= h_rgb - 1)
        )
        valid &= in_b
        d_obs = depth_views_f32[v, np.clip(v_di, 0, h_d - 1), np.clip(u_di, 0, w_d - 1)].astype(np.float64)
        valid &= np.isfinite(d_obs) & (d_obs > 0.0) & (np.abs(z - d_obs) <= depth_tol_m)

        diff = cam_o_f32[v].astype(np.float64)[None, :] - xyz
        dist_sq = np.sum(diff * diff, axis=1)
        view_dir = diff / np.sqrt(np.maximum(dist_sq, 1e-12))[:, None]
        inv_dist_sq = 1.0 / np.maximum(dist_sq, 0.25)

        if unit_n is not None and has_n is not None:
            cos_a = np.sum(unit_n * view_dir, axis=1)
            cos_a = np.where(has_n, cos_a, 1.0)
            valid &= cos_a >= min_cos_angle
            w = np.where(valid, cos_a * inv_dist_sq, 0.0)
        else:
            w = np.where(valid, inv_dist_sq, 0.0)

        idx = np.nonzero(valid)[0]
        if len(idx) == 0:
            continue

        # Bilinear RGB
        uc = np.clip(u_rgb[idx], 0.0, w_rgb - 1.0)
        vc = np.clip(v_rgb[idx], 0.0, h_rgb - 1.0)
        x0 = np.floor(uc).astype(np.int64)
        y0 = np.floor(vc).astype(np.int64)
        x1 = np.minimum(x0 + 1, w_rgb - 1)
        y1 = np.minimum(y0 + 1, h_rgb - 1)
        wx = (uc - x0)[:, None]
        wy = (vc - y0)[:, None]
        img = rgb_views_u8[v, :, :, :3].astype(np.float64) / 255.0
        c00 = img[y0, x0]
        c10 = img[y0, x1]
        c01 = img[y1, x0]
        c11 = img[y1, x1]
        rgb_obs = (1 - wy) * ((1 - wx) * c00 + wx * c10) + wy * ((1 - wx) * c01 + wx * c11)
        y_obs = rgb_obs - 0.5

        vd = view_dir[idx]
        if sh_degree == 0:
            basis = np.full((len(idx), 1), SH_C0, dtype=np.float64)
        else:
            basis = np.stack(
                [np.full(len(idx), SH_C0), -SH_C1 * vd[:, 1], SH_C1 * vd[:, 2], -SH_C1 * vd[:, 0]],
                axis=-1,
            )
        wi = w[idx]
        wb = basis * wi[:, None]
        for k, p_i in enumerate(idx):
            atwa[p_i] += np.outer(wb[k], basis[k])
            atwy[p_i] += np.outer(wb[k], y_obs[k])
            rgb_wsum[p_i] += wi[k] * rgb_obs[k]
            weight_sum[p_i] += wi[k]
            obs_count[p_i] += 1
        records.append((idx, rgb_obs, wi, basis))

    observed = (obs_count > 0) & (weight_sum > 0.0)
    mean_rgb = np.clip(rgb_wsum / np.maximum(weight_sum, 1e-12)[:, None], 0.0, 1.0)
    fb = np.clip(fallback_rgb_f32.astype(np.float64), 0.0, 1.0)
    rgb_init = np.where(observed[:, None], mean_rgb, fb)

    c_prior = np.zeros((n_pts, n_basis, 3), dtype=np.float64)
    c_prior[:, 0, :] = (rgb_init - 0.5) / SH_C0

    reg_eff = max(float(reg_lambda), 1e-8)
    lhs = atwa + reg_eff * np.eye(n_basis, dtype=np.float64)[None, :, :]
    rhs = atwy + float(reg_lambda) * c_prior
    sol = np.linalg.solve(lhs, rhs)
    sh_active = np.where((obs_count >= 2)[:, None, None], sol, c_prior)

    sh_full = np.zeros((n_pts, 16, 3), dtype=np.float32)
    sh_full[:, :n_basis, :] = sh_active.astype(np.float32)

    eval_mask = (obs_count >= 2) if np.any(obs_count >= 2) else (obs_count > 0)
    se_num = 0.0
    se_den = 0.0
    for idx, rgb_obs, wi, basis in records:
        m = eval_mask[idx]
        if not np.any(m):
            continue
        pred = np.clip(0.5 + np.einsum("mb,mbc->mc", basis[m], sh_active[idx[m]]), 0.0, 1.0)
        mse = np.mean((pred - rgb_obs[m]) ** 2, axis=-1)
        se_num += float(np.sum(wi[m] * mse))
        se_den += float(np.sum(wi[m]))
    rmse = float(np.sqrt(max(se_num, 0.0) / se_den)) if se_den > 0.0 else 0.0

    return {
        "sh_coeffs": sh_full,
        "rgb_init": rgb_init.astype(np.float32),
        "obs_count": obs_count,
        "weight_sum": weight_sum.astype(np.float32),
        "wls_sh_fit_rmse": rmse,
    }


def _build_synthetic_multiview(n_points: int = 512, n_views: int = 8, h: int = 64, w: int = 80, seed: int = 7):
    rng = np.random.default_rng(seed)
    # Points on a sphere/patch around origin + some far/unobserved points
    means = rng.uniform(-0.8, 0.8, size=(n_points, 3)).astype(np.float32)
    means[:16] = np.array([10.0, 10.0, 10.0], dtype=np.float32)  # guaranteed unobserved
    normals = rng.normal(size=(n_points, 3)).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    normals[16:32] = 0.0  # zero-length normal fallback

    k_mat = np.array([[w * 0.9, 0.0, w * 0.5], [0.0, h * 0.9, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    k_rgb = np.repeat(k_mat[None, :, :], n_views, axis=0)
    k_depth = k_rgb.copy()

    w2c_list = []
    cam_o_list = []
    for v in range(n_views):
        ang = 2.0 * np.pi * v / n_views
        cam_pos = np.array([2.5 * np.cos(ang), 0.4 * np.sin(2.0 * ang), 2.5 * np.sin(ang)], dtype=np.float64)
        c2w = _make_lookat_c2w(cam_pos, np.zeros(3, dtype=np.float64))
        w2c_list.append(np.linalg.inv(c2w).astype(np.float32))
        cam_o_list.append(cam_pos.astype(np.float32))
    w2c = np.stack(w2c_list, axis=0)
    cam_o = np.stack(cam_o_list, axis=0)

    # Smooth view-dependent RGB + depth consistent with nearest projected points
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb_views = np.zeros((n_views, h, w, 3), dtype=np.uint8)
    depth_views = np.full((n_views, h, w), 2.5, dtype=np.float32)
    for v in range(n_views):
        r_ch = np.clip(0.3 + 0.5 * (xx / max(w - 1, 1)) + 0.1 * np.sin(v), 0.0, 1.0)
        g_ch = np.clip(0.2 + 0.6 * (yy / max(h - 1, 1)) + 0.1 * np.cos(v), 0.0, 1.0)
        b_ch = np.clip(0.4 + 0.3 * np.sin(0.1 * xx + v), 0.0, 1.0)
        rgb_views[v] = np.round(np.stack([r_ch, g_ch, b_ch], axis=-1) * 255.0).astype(np.uint8)
        # Set depth map from projected points so many points pass depth check and some fail
        p_cam = means @ w2c[v, :3, :3].T + w2c[v, :3, 3][None, :]
        z = p_cam[:, 2]
        u = np.rint(k_depth[v, 0, 0] * (p_cam[:, 0] / np.maximum(z, 0.2)) + k_depth[v, 0, 2]).astype(np.int64)
        vv = np.rint(k_depth[v, 1, 1] * (p_cam[:, 1] / np.maximum(z, 0.2)) + k_depth[v, 1, 2]).astype(np.int64)
        ok = (z > 0.2) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        for pi in np.nonzero(ok)[0]:
            depth_views[v, vv[pi], u[pi]] = z[pi]

    fallback = rng.uniform(0.1, 0.9, size=(n_points, 3)).astype(np.float32)
    return means, normals, rgb_views, depth_views, k_rgb, k_depth, w2c, cam_o, fallback


@pytest.mark.parametrize("sh_degree", [1, 0])
@pytest.mark.parametrize("with_normals", [True, False])
def test_project_and_fit_sh_vulkan_parity(tmp_path, sh_degree, with_normals):
    ds = sc.build_dataset(tmp_path / "ds", n_points=64, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out")

    means, normals, rgb_views, depth_views, k_rgb, k_depth, w2c, cam_o, fallback = _build_synthetic_multiview(
        n_points=512, n_views=8, h=64, w=80, seed=11 + sh_degree + int(with_normals)
    )
    n_in = normals if with_normals else None

    mod.reset_gpu_stage_timestamps()
    out_vk = mod.project_and_fit_sh_vulkan(
        means, n_in, rgb_views, depth_views, k_rgb, k_depth, w2c, cam_o, fallback,
        sh_degree=sh_degree, reg_lambda=0.25, depth_tol_m=0.12, min_cos_angle=0.15,
    )
    ref = _cpu_reference_project_and_fit_sh(
        means, n_in, rgb_views, depth_views, k_rgb, k_depth, w2c, cam_o, fallback,
        sh_degree=sh_degree, reg_lambda=0.25, depth_tol_m=0.12, min_cos_angle=0.15,
    )

    assert out_vk["backend"] == "vulkan_slang"
    assert out_vk["sh_coeffs"].shape == (len(means), 16, 3)
    assert out_vk["rgb_init"].shape == (len(means), 3)
    assert out_vk["obs_count"].shape == (len(means),)
    assert out_vk["weight_sum"].shape == (len(means),)
    assert int(np.sum(out_vk["obs_count"] >= 2)) > 50
    assert int(np.sum(out_vk["obs_count"] == 0)) >= 16

    obs_diff = int(np.max(np.abs(out_vk["obs_count"] - ref["obs_count"])))
    sh_mae = float(np.mean(np.abs(out_vk["sh_coeffs"] - ref["sh_coeffs"])))
    rgb_mae = float(np.mean(np.abs(out_vk["rgb_init"] - ref["rgb_init"])))
    w_mae = float(np.mean(np.abs(out_vk["weight_sum"] - ref["weight_sum"])))
    rmse_diff = abs(float(out_vk["wls_sh_fit_rmse"]) - float(ref["wls_sh_fit_rmse"]))

    assert obs_diff == 0
    assert sh_mae < 1e-3, f"sh_mae={sh_mae}"
    assert rgb_mae < 1e-4, f"rgb_mae={rgb_mae}"
    assert w_mae < 1e-4, f"w_mae={w_mae}"
    assert rmse_diff < 1e-4, f"rmse_diff={rmse_diff}"
    assert out_vk["wls_sh_fit_rmse"] > 0.0
    assert out_vk["kernel_wall_ms"] > 0.0
    assert out_vk["total_wall_ms"] >= out_vk["kernel_wall_ms"]

    ts = mod.get_gpu_stage_timestamps_ms()
    assert isinstance(ts["timestamp_queries_enabled"], bool)
    assert ts["appear_init_ms"] > 0.0
    assert ts["total_gpu_ms"] >= ts["appear_init_ms"]


def test_gpu_stage_timestamps_and_reset(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=192, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)

    mod.reset_gpu_stage_timestamps()
    ts0 = mod.get_gpu_stage_timestamps_ms()
    for k in ("forward_ms", "backward_ms", "optimizer_ms", "rayguard_ms", "appear_init_ms", "total_gpu_ms"):
        assert ts0[k] == 0.0

    # Run a few training steps + RayGuard observation
    for s in range(1, 5):
        mod.train_step(s % int(mod.num_train), s)

    depth = np.full((48, 64), 3.0, dtype=np.float32)
    n2c = np.eye(3, 4, dtype=np.float32)
    n2c[2, 3] = 2.0
    cam_o = np.array([0.0, 0.0, -2.0], dtype=np.float32)
    mod.observe_and_apply_rayguard_gpu(depth, 50.0, 50.0, 32.0, 24.0, n2c, cam_o, 1.0, 5)

    ts1 = mod.get_gpu_stage_timestamps_ms()
    assert ts1["timestamp_queries_enabled"] is True
    assert ts1["forward_ms"] > 0.0
    assert ts1["backward_ms"] > 0.0
    assert ts1["optimizer_ms"] > 0.0
    assert ts1["rayguard_ms"] > 0.0
    expected_total = (
        ts1["forward_ms"] + ts1["backward_ms"] + ts1["optimizer_ms"]
        + ts1["rayguard_ms"] + ts1["appear_init_ms"]
    )
    assert abs(ts1["total_gpu_ms"] - expected_total) < 1e-6

    mod.reset_gpu_stage_timestamps()
    ts2 = mod.get_gpu_stage_timestamps_ms()
    assert ts2["total_gpu_ms"] == 0.0


def test_compact_mode0_active_set_and_frozen_skip(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=200, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    mod.set_freeze_skip_backward(True)

    n = int(len(mod.xyz_ws))
    assert n == 200
    gids = np.full((n,), -1, dtype=np.int32)
    gids[:50] = 2  # LOCKED structural group (Mode-0 frozen)
    mod.set_splat_group_ids(gids)
    mod.set_group_state(2, 2)  # LOCKED

    states = np.full((n,), 5, dtype=np.uint32)  # MEASURED_FREE
    states[:50] = 0xFFFFFFFF  # LEGACY_GROUP -> inherits LOCKED group state
    states[50:70] = 4  # CONTRADICTED -> Mode-0 frozen
    mod.set_splat_states(states)

    stats = mod.compact_mode0_active_set()
    assert stats["n_total"] == 200
    assert stats["n_frozen_mode0"] == 70
    assert stats["n_active_trainable"] == 130
    assert abs(stats["active_fraction"] - 130.0 / 200.0) < 1e-9

    xyz_before = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_before = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_before = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    for s in range(1, 8):
        mod.train_step(s % int(mod.num_train), s)

    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_after = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_after = np.asarray(mod.rotations, dtype=np.float32)
    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)

    # Frozen Mode-0 splats ([:70]) must be completely untouched by backward & optimizer
    assert np.array_equal(xyz_after[:70], xyz_before[:70])
    assert np.array_equal(sh_after[:70], sh_before[:70])
    assert np.array_equal(rot_after[:70], rot_before[:70])
    assert np.array_equal(so_after[:70], so_before[:70])

    # Active trainable splats ([70:]) must have updated
    assert float(np.max(np.abs(xyz_after[70:] - xyz_before[70:]))) > 1e-6
    assert float(np.max(np.abs(sh_after[70:] - sh_before[70:]))) > 1e-6

    # Forward rendering still runs cleanly with frozen Mode-0 splats present
    mod.render_train(0)
    pix = np.asarray(mod.pixel_state, dtype=np.float32)
    assert np.all(np.isfinite(pix))
