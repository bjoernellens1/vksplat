"""Tests for vksplat PR #19:
Track A (AGO-0 Telemetry Semantic Fixes):
  1. True alpha-composited visibility contrib_sum = sum(T * alpha): visible foreground
     Gaussians get contrib_sum > 0, while offscreen and fully occluded high-opacity
     Gaussians get contrib_sum == 0.0.
  2. True per-pixel AbsGS abs_proj_grad = sum(length(dL/dmean2d_pixel)), full SH
     (sh0 + shN) in predicted_gain / grad_norm / update_norm, post-normalization quat
     update, and momentum-aware update_count.
Track B (Native Differentiable Physical Expected-Depth Loss D1):
  3. Zero perturbation (< 1e-6) when lambda_depth == 0.0 even with depth maps attached.
  4. Exact central finite-difference gradient validation for both dL/dz_i and dL/dalpha_i
     (single-Gaussian and overlapping two-Gaussian fixtures) + monotonic z convergence.
P0 Addition (Explicit Optimization Active-Set API):
  5. set_active_set / get_active_set / clear_active_set: excluded Gaussians AND their
     Adam moments remain bit-identical (max_abs_diff == 0.0), clear_active_set restores
     full-map updates, and snapshot_ago_state/restore_ago_state round-trips active_set.
"""
from __future__ import annotations

from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

sys.path.insert(0, str(sc.ROOT))
import vksplat  # noqa: E402
from simple_trainer import TrainerConfig, join_dir  # noqa: E402


def _world_to_norm_points(dp_row_major: np.ndarray, xyz_world: np.ndarray) -> tuple[np.ndarray, float]:
    dp = np.asarray(dp_row_major, dtype=np.float64)
    scale = float(np.linalg.norm(dp[:3, 0]))
    xyz_norm = (xyz_world.astype(np.float64) @ dp[:3, :3].T + dp[:3, 3][None, :]).astype(np.float32)
    return xyz_norm, scale


def _norm_to_world_points(dp_row_major: np.ndarray, xyz_norm: np.ndarray) -> np.ndarray:
    dp_inv = np.linalg.inv(np.asarray(dp_row_major, dtype=np.float64))
    return (xyz_norm.astype(np.float64) @ dp_inv[:3, :3].T + dp_inv[:3, 3][None, :]).astype(np.float32)


def _huber_depth_loss(
    depth_exp_m: np.ndarray,
    alpha: np.ndarray,
    depth_sensor_m: np.ndarray,
    weight_map: np.ndarray,
    lambda_depth: float,
    huber_delta: float,
) -> float:
    h, w = depth_exp_m.shape
    valid = (depth_sensor_m > 0.0) & (weight_map > 0.0) & (alpha > 0.1)
    if not np.any(valid):
        return 0.0
    r = (depth_exp_m[valid] - depth_sensor_m[valid]).astype(np.float64)
    abs_r = np.abs(r)
    d = float(huber_delta)
    loss_pix = np.where(abs_r <= d, 0.5 * (r * r) / d, abs_r - 0.5 * d)
    return float(lambda_depth * np.sum(weight_map[valid].astype(np.float64) * loss_pix) / float(h * w))


def _make_single_view_trainer(
    tmp_path: Path,
    name: str,
    xyz_world: np.ndarray,
    scales_world: np.ndarray,
    opacities: np.ndarray,
    rgb_init: np.ndarray,
    target_rgba: np.ndarray,
    w: int = 64,
    h: int = 48,
    fx: float = 60.0,
    fy: float = 60.0,
):
    """Create a trainer with a single identity-pose camera looking along +Z and exact Gaussians."""
    ds = sc.build_dataset(tmp_path / f"ds_{name}", n_points=16, w=w, h=h, structured=True)
    out_dir = tmp_path / f"out_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = vksplat.VkSplat()
    mod.initialize(join_dir(str(sc.ROOT), "shader"), -1)
    cfg = TrainerConfig(
        dataset_dir=str(ds),
        image_dir=join_dir(str(ds), "images"),
        mask_dir="",
        sparse_dir=join_dir(str(ds), "sparse/0"),
        output_dir=str(out_dir),
        eval_interval=100000,
        refine_start_iter=100000,
        refine_stop_iter=0,
        train_steps=50,
        max_steps=500,
        enable_viewer=False,
        save_train_renders=False,
    )
    train_meta = mod.set_train_config(asdict(cfg))
    dp = np.asarray(train_meta["dataparser_transform"], dtype=np.float32)
    xyz_norm, dp_scale = _world_to_norm_points(dp, xyz_world)
    scales_norm = (scales_world * dp_scale).astype(np.float32)
    n = int(xyz_world.shape[0])

    # Replace training frames with a single clean identity-pose camera
    mod.clear_train_frames()
    c2w = np.eye(4, dtype=np.float32)
    mod.append_train_frame(
        np.ascontiguousarray(target_rgba, dtype=np.uint8),
        c2w,
        fx,
        fy,
        w * 0.5,
        h * 0.5,
        "synth_0.png",
    )

    # Convert RGB in [0,1] to SH DC (c = 0.5 + C0 * sh0 -> sh0 = (c - 0.5) / C0)
    c0 = 0.28209479177387814
    sh = np.zeros((n, 16, 3), dtype=np.float32)
    sh[:, 0, :] = (rgb_init.astype(np.float32) - 0.5) / c0
    rot = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    so = np.concatenate([scales_norm, opacities.astype(np.float32)[:, None]], axis=1).astype(np.float32)

    snap = {
        "num_gaussians": n,
        "step": 0,
        "xyz_ws": xyz_norm,
        "sh_coeffs": sh,
        "rotations": rot,
        "scales_opacs": so,
        "has_optimizer_moments": False,
    }
    mod.restore_ago_state(snap)
    return mod, dp, dp_scale


@pytest.mark.parametrize("bwd_variant", [1, 3])
def test_true_visibility_contrib_sum_and_absgs_per_pixel(tmp_path, bwd_variant):
    """Visible foreground Gaussian gets contrib_sum > 0 and abs_proj_grad > 0;
    offscreen and fully occluded high-opacity Gaussians get contrib_sum == 0.0."""
    w, h = 64, 48
    # Gaussians 0, 1, 2: three dense foreground Gaussians at z=2.0 (0.99 opacity -> T < 1e-4 across center)
    # Gaussian 3: occluded directly behind them at z=3.5 with high opacity=0.95
    # Gaussian 4: far offscreen at x=50.0, z=2.0 with high opacity=0.95
    xyz_world = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.01],
        [0.0, 0.0, 2.02],
        [0.0, 0.0, 3.5],
        [50.0, 0.0, 2.0],
    ], dtype=np.float32)
    scales_world = np.array([
        [0.60, 0.60, 0.05],
        [0.60, 0.60, 0.05],
        [0.60, 0.60, 0.05],
        [0.05, 0.05, 0.05],
        [0.30, 0.30, 0.05],
    ], dtype=np.float32)
    opacities = np.array([0.99, 0.99, 0.99, 0.95, 0.95], dtype=np.float32)
    rgb_init = np.array([
        [0.2, 0.3, 0.8],
        [0.2, 0.3, 0.8],
        [0.2, 0.3, 0.8],
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
    ], dtype=np.float32)
    target_rgba = np.full((h, w, 4), [220, 80, 40, 255], dtype=np.uint8)

    mod, _, _ = _make_single_view_trainer(
        tmp_path, f"vis_v{bwd_variant}", xyz_world, scales_world, opacities, rgb_init, target_rgba, w=w, h=h
    )
    mod.force_backward_variant(bwd_variant)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()

    mod.train_step(0, 0)
    tel = mod.get_ago_telemetry()

    # Visible foreground Gaussian 0: strictly positive visibility, AbsGS grad, and predicted gain
    assert float(tel["contrib_sum"][0]) > 1.0, f"Foreground contrib_sum too small: {tel['contrib_sum']}"
    assert float(tel["abs_proj_grad"][0]) > 0.0, f"Foreground abs_proj_grad == 0: {tel['abs_proj_grad']}"
    assert float(tel["predicted_gain"][0]) > 0.0
    assert float(tel["update_count"][0]) == 1.0

    # Offscreen high-opacity Gaussian 4: MUST have exactly 0.0 visibility & updates
    assert float(tel["contrib_sum"][4]) == 0.0, f"Offscreen Gaussian got non-zero contrib_sum: {tel['contrib_sum'][4]}"
    assert float(tel["abs_proj_grad"][4]) == 0.0
    assert float(tel["update_count"][4]) == 0.0

    # Fully occluded Gaussian 3 behind three 0.99-opacity splats: T < 0.0001 -> early-terminated -> 0.0!
    assert float(tel["contrib_sum"][3]) == 0.0, (
        f"Occluded Gaussian got non-zero contrib_sum: {tel['contrib_sum'][3]} "
        f"(foreground={tel['contrib_sum'][0]})"
    )

    mod.cleanup()


def test_absgs_triangle_inequality_and_full_sh_telemetry(tmp_path):
    """Verify abs_proj_grad >= ||net v_xy_vs|| (triangle inequality) and full SH telemetry."""
    ds = sc.build_dataset(tmp_path / "ds_absgs", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_absgs", refine_start_iter=100000, refine_stop_iter=0)
    mod.force_backward_variant(1)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()

    # Run at step=1000 so active_sh=1 (both sh0 and shN active!)
    mod.train_step(0, 1000)
    tel = mod.get_ago_telemetry()
    v_xy = np.asarray(mod.v_xy_vs, dtype=np.float32)
    net_xy_norm = np.linalg.norm(v_xy, axis=1)

    updated = tel["update_count"] >= 1.0
    assert int(np.sum(updated)) > 50
    # Triangle inequality: sum_p ||g_p|| >= ||sum_p g_p|| for every updated Gaussian
    assert np.all(tel["abs_proj_grad"][updated] + 1e-6 >= net_xy_norm[updated]), (
        f"AbsGS triangle inequality violated: min diff = "
        f"{float(np.min(tel['abs_proj_grad'][updated] - net_xy_norm[updated]))}"
    )
    # And for at least some Gaussians with multi-pixel cancellation, abs_proj_grad strictly exceeds net norm
    assert int(np.sum(tel["abs_proj_grad"][updated] > net_xy_norm[updated] + 1e-6)) > 10
    assert np.all(tel["contrib_sum"][updated] > 0.0)
    assert np.all(tel["predicted_gain"][updated] > 0.0)

    mod.cleanup()


def test_depth_loss_zero_perturbation_when_disabled(tmp_path):
    """With depth maps attached but lambda_depth == 0.0, training 5 steps matches
    baseline without depth maps to < 1e-6."""
    ds = sc.build_dataset(tmp_path / "ds_d0", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_d0")
    mod.force_backward_variant(1)

    assert mod.get_depth_loss_weight() == 0.0

    for s in range(1, 3):
        mod.train_step(s % int(mod.num_train), s)
    snap = mod.snapshot_ago_state()

    # Branch A: no depth attached
    for s in range(3, 8):
        mod.train_step(s % int(mod.num_train), s)
    xyz_a = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_a = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    so_a = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    # Branch B: restore, attach depth maps to all frames, keep lambda_depth=0.0
    mod.restore_ago_state(snap)
    depth_m = np.full((48, 64), 2.5, dtype=np.float32)
    for fi in range(int(mod.num_train)):
        mod.set_train_depth(fi, depth_m)
    mod.set_depth_loss_weight(0.0, 0.15)
    assert mod.get_depth_loss_weight() == 0.0
    for s in range(3, 8):
        mod.train_step(s % int(mod.num_train), s)
    xyz_b = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_b = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    so_b = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    assert float(np.max(np.abs(xyz_a - xyz_b))) < 1e-6
    assert float(np.max(np.abs(sh_a - sh_b))) < 1e-6
    assert float(np.max(np.abs(so_a - so_b))) < 1e-6

    mod.clear_train_depths()
    mod.cleanup()


@pytest.mark.parametrize("bwd_variant", [1, 3])
def test_depth_loss_finite_difference_gradient_and_convergence(tmp_path, bwd_variant):
    """Validate analytical D1 expected-depth gradients (both dL/dz and dL/dalpha) against
    central finite differences and verify monotonic z convergence toward target depth_m."""
    w, h = 64, 48
    fx = fy = 60.0
    target_depth_m = 2.0
    lambda_depth = 2.0
    huber_delta = 0.5  # quadratic regime for |r| <= 0.5 so derivative is smooth & linear
    c2w = np.eye(4, dtype=np.float32)
    depth_map = np.full((h, w), target_depth_m, dtype=np.float32)
    weight_map = np.ones((h, w), dtype=np.float32)
    target_rgba = np.full((h, w, 4), [128, 128, 128, 255], dtype=np.uint8)
    eps_z = 2e-3

    # --- Part A: Exact 1-Gaussian dL/dz_world check (broad Gaussian covering viewport so A > 0.1 everywhere) ---
    xyz_1 = np.array([[0.0, 0.0, 2.5]], dtype=np.float32)
    sc_1 = np.array([[3.0, 3.0, 0.05]], dtype=np.float32)
    op_1 = np.array([0.85], dtype=np.float32)
    rgb_1 = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    mod1, dp1, dp_scale1 = _make_single_view_trainer(
        tmp_path, f"fd1_v{bwd_variant}", xyz_1, sc_1, op_1, rgb_1, target_rgba, w=w, h=h, fx=fx, fy=fy
    )
    mod1.force_backward_variant(bwd_variant)
    mod1.set_train_depth(0, depth_map, weight_map)
    mod1.set_depth_loss_weight(lambda_depth, huber_delta)
    snap1 = mod1.snapshot_ago_state()

    def _eval_loss1(z_val: float) -> float:
        sm = dict(snap1)
        zw = np.array([[0.0, 0.0, z_val]], dtype=np.float32)
        sm["xyz_ws"], _ = _world_to_norm_points(dp1, zw)
        mod1.restore_ago_state(sm)
        rgbd = mod1.render_pose_rgbd(c2w, fx, fy, w * 0.5, h * 0.5, w, h)
        return _huber_depth_loss(
            rgbd["depth_expected"], rgbd["alpha"], depth_map, weight_map, lambda_depth, huber_delta
        )

    mod1.restore_ago_state(snap1)
    mod1.train_step(0, 0)
    an_g1 = float(np.asarray(mod1.v_depths, dtype=np.float32).reshape(-1)[0]) * dp_scale1
    fd_g1 = (_eval_loss1(2.5 + eps_z) - _eval_loss1(2.5 - eps_z)) / (2.0 * eps_z)
    rel_err1 = abs(an_g1 - fd_g1) / max(abs(fd_g1), 1e-6)
    print(f"[variant={bwd_variant}] 1-Gaussian analytical={an_g1:.6f} vs FD={fd_g1:.6f} (rel_err={rel_err1:.4e})")
    assert an_g1 > 0.0 and fd_g1 > 0.0
    assert rel_err1 < 1e-2, f"1-Gaussian FD mismatch: analytical={an_g1}, fd={fd_g1}, rel_err={rel_err1}"

    # --- Part B: Overlapping 2-Gaussian fixture (validates BOTH dL/dz_i AND dL/dalpha_i!) ---
    # Use exact black RGB (0.0) and black target (0) so RGB L1+SSIM gradient is identically 0.0!
    xyz_world = np.array([
        [0.0, 0.0, 2.25],
        [0.0, 0.0, 2.65],
    ], dtype=np.float32)
    scales_world = np.array([
        [3.0, 3.0, 0.05],
        [3.0, 3.0, 0.05],
    ], dtype=np.float32)
    opacities = np.array([0.55, 0.65], dtype=np.float32)
    rgb_zero = np.zeros((2, 3), dtype=np.float32)
    target_zero = np.full((h, w, 4), [0, 0, 0, 255], dtype=np.uint8)

    mod, dp, dp_scale = _make_single_view_trainer(
        tmp_path, f"fd2_v{bwd_variant}", xyz_world, scales_world, opacities, rgb_zero, target_zero,
        w=w, h=h, fx=fx, fy=fy,
    )
    mod.force_backward_variant(bwd_variant)
    mod.set_train_depth(0, depth_map, weight_map)
    mod.set_depth_loss_weight(lambda_depth, huber_delta)
    base_snap = mod.snapshot_ago_state()

    def _eval_depth_loss(z_w: np.ndarray, opacs: np.ndarray) -> float:
        snap_mod = dict(base_snap)
        xyz_n, _ = _world_to_norm_points(dp, z_w)
        so = np.asarray(base_snap["scales_opacs"], dtype=np.float32).copy()
        so[:, 3] = opacs.astype(np.float32)
        snap_mod["xyz_ws"] = xyz_n
        snap_mod["scales_opacs"] = so
        mod.restore_ago_state(snap_mod)
        rgbd = mod.render_pose_rgbd(c2w, fx, fy, w * 0.5, h * 0.5, w, h)
        return _huber_depth_loss(
            rgbd["depth_expected"], rgbd["alpha"], depth_map, weight_map, lambda_depth, huber_delta
        )

    # Run 1 step to populate pure depth-loss v_depths and v_inv_cov_vs_opacity
    mod.restore_ago_state(base_snap)
    mod.train_step(0, 0)
    grad_z_analytical = np.asarray(mod.v_depths, dtype=np.float32).reshape(-1).copy() * dp_scale
    an_grad_op0 = float(np.asarray(mod.v_inv_cov_vs_opacity, dtype=np.float32)[0, 3])

    fd_grad_z = np.zeros(2, dtype=np.float64)
    for gi in range(2):
        z_plus = xyz_world.copy()
        z_minus = xyz_world.copy()
        z_plus[gi, 2] += eps_z
        z_minus[gi, 2] -= eps_z
        fd_grad_z[gi] = (_eval_depth_loss(z_plus, opacities) - _eval_depth_loss(z_minus, opacities)) / (2.0 * eps_z)

    print(
        f"[variant={bwd_variant}] 2-Gaussian analytical dL/dz_world={grad_z_analytical.tolist()} "
        f"vs central FD={fd_grad_z.tolist()}"
    )
    for gi in range(2):
        assert np.sign(grad_z_analytical[gi]) == np.sign(fd_grad_z[gi]) and grad_z_analytical[gi] > 0.0
        rel_err = abs(float(grad_z_analytical[gi]) - float(fd_grad_z[gi])) / max(abs(float(fd_grad_z[gi])), 1e-6)
        assert rel_err < 0.05, (
            f"2-Gaussian dL/dz_world mismatch for Gaussian {gi}: analytical={grad_z_analytical[gi]:.6f}, "
            f"fd={fd_grad_z[gi]:.6f}, rel_err={rel_err:.4f}"
        )

    # Exact dL/dopacity_0 check (validates alpha-compositing backward dL/dalpha_i!)
    eps_o = 2e-3
    op_p = opacities.copy(); op_p[0] += eps_o
    op_m = opacities.copy(); op_m[0] -= eps_o
    fd_grad_op0 = (_eval_depth_loss(xyz_world, op_p) - _eval_depth_loss(xyz_world, op_m)) / (2.0 * eps_o)
    rel_err_op0 = abs(an_grad_op0 - fd_grad_op0) / max(abs(fd_grad_op0), 1e-6)
    print(f"[variant={bwd_variant}] 2-Gaussian analytical dL/dopac_0={an_grad_op0:.6f} vs FD={fd_grad_op0:.6f} (rel_err={rel_err_op0:.4e})")
    assert an_grad_op0 < 0.0 and fd_grad_op0 < 0.0
    assert rel_err_op0 < 0.02, f"dL/dopac_0 mismatch: analytical={an_grad_op0}, fd={fd_grad_op0}, rel_err={rel_err_op0}"

    # --- Part C: Monotonic convergence of z from 2.5m toward 2.0m ---
    mod1.restore_ago_state(snap1)
    z_history = [float(_norm_to_world_points(dp1, np.asarray(mod1.xyz_ws, dtype=np.float32))[0, 2])]
    for s in range(25):
        mod1.train_step(0, s)
        z_now = float(_norm_to_world_points(dp1, np.asarray(mod1.xyz_ws, dtype=np.float32))[0, 2])
        z_history.append(z_now)

    assert z_history[-1] < z_history[0] - 0.01, (
        f"Expected z to move significantly toward 2.0m from 2.5m, got {z_history[0]:.4f} -> {z_history[-1]:.4f}"
    )
    for k in range(1, len(z_history)):
        assert z_history[k] <= z_history[k - 1] + 1e-5, (
            f"Non-monotonic z movement at step {k}: {z_history[k-1]:.5f} -> {z_history[k]:.5f}"
        )

    mod1.cleanup()
    mod.cleanup()


def test_explicit_active_set_api_and_snapshot_roundtrip(tmp_path):
    """set_active_set updates ONLY selected Gaussians while excluded Gaussians AND their
    Adam moments remain bit-identical (max_abs_diff == 0.0); clear_active_set restores
    full-map updates; snapshot_ago_state/restore_ago_state preserves active_set cleanly."""
    ds = sc.build_dataset(tmp_path / "ds_act", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_act", refine_start_iter=100000, refine_stop_iter=0)
    mod.force_backward_variant(1)

    n = int(len(mod.xyz_ws))
    assert n == 400
    assert len(mod.get_active_set()) == 0

    # Warm up 2 steps so all Adam moments are allocated and non-zero
    for s in range(0, 2):
        mod.train_step(s % int(mod.num_train), s)

    # Validation: out-of-range index must raise
    with pytest.raises((IndexError, ValueError, RuntimeError)):
        mod.set_active_set(np.array([0, n], dtype=np.uint32))

    # Set explicit active set to first 128 Gaussians (multiple of subgroup/SH reorder size 32 & 64)
    k_act = 128
    subset = np.arange(0, k_act, dtype=np.uint32)
    mod.set_active_set(subset)
    got = mod.get_active_set()
    assert got.dtype == np.uint32
    assert np.array_equal(got, subset)

    # Snapshot parameters and Adam moments before stepping
    xyz_before = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_before = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_before = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()
    mom_before = {k: np.asarray(v, dtype=np.float32).copy() for k, v in mod.get_optimizer_moments().items()}

    # Take AGO snapshot with active set attached
    snap_with_active = mod.snapshot_ago_state()

    # Step 4 times with active set = [0..127]
    for s in range(2, 6):
        mod.train_step(s % int(mod.num_train), s)

    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_after = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_after = np.asarray(mod.rotations, dtype=np.float32)
    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)
    mom_after = {k: np.asarray(v, dtype=np.float32) for k, v in mod.get_optimizer_moments().items()}

    # Excluded Gaussians [128:400]: bit-identical parameters
    assert float(np.max(np.abs(xyz_after[k_act:] - xyz_before[k_act:]))) == 0.0
    assert float(np.max(np.abs(sh_after[k_act:] - sh_before[k_act:]))) == 0.0
    assert float(np.max(np.abs(rot_after[k_act:] - rot_before[k_act:]))) == 0.0
    assert float(np.max(np.abs(so_after[k_act:] - so_before[k_act:]))) == 0.0

    # Excluded Gaussians [128:400]: bit-identical Adam moments (flat buffer offsets align on 128-splat boundary)
    mom_strides = {
        "g_xyz_ws": 6,
        "g_rotations": 8,
        "g_scales_opacs": 8,
        "g_sh_coeffs_1": 48,
        "g_sh_coeffs_2": 48,
    }
    for k, stride in mom_strides.items():
        assert k in mom_before and k in mom_after
        off = k_act * stride
        assert float(np.max(np.abs(mom_after[k][off:] - mom_before[k][off:]))) == 0.0, (
            f"Excluded Gaussian Adam moment {k} mutated!"
        )

    # Active Gaussians [0:128]: MUST have updated
    assert float(np.max(np.abs(xyz_after[:k_act] - xyz_before[:k_act]))) > 1e-6
    assert float(np.max(np.abs(sh_after[:k_act] - sh_before[:k_act]))) > 1e-6

    # Verify snapshot_ago_state / restore_ago_state preserves active_set
    mod.clear_active_set()
    assert len(mod.get_active_set()) == 0
    mod.restore_ago_state(snap_with_active)
    assert np.array_equal(mod.get_active_set(), subset)

    # Verify clear_active_set restores full-map updates
    mod.clear_active_set()
    assert len(mod.get_active_set()) == 0
    xyz_c0 = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    for s in range(2, 5):
        mod.train_step(s % int(mod.num_train), s)
    xyz_c1 = np.asarray(mod.xyz_ws, dtype=np.float32)
    assert float(np.max(np.abs(xyz_c1[k_act:] - xyz_c0[k_act:]))) > 1e-6

    mod.cleanup()
