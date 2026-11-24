"""Tests for vksplat PR #20 — GPU-Resident Zero-Readback AGO-2 Active-Set Selection:
1. Exactness & bit-identical exclusion: select_ago_active_set_gpu matches CPU reference
   indices from get_ago_telemetry(), respects frozen states, falls back cleanly on empty
   selection, and leaves excluded Gaussians bit-identical across subsequent train_step().
2. Newborn protection: Gaussians with update_count < newborn_warmup_steps are always
   included even under a huge gain_threshold (unless frozen with exclude_frozen=True).
3. Adaptive selection & 16-byte summary: select_ago_adaptive_active_set_gpu() and
   get_ago_step_gain_summary() return finite, consistent scalars.
4. Latency benchmark: 20 calls to select_ago_adaptive_active_set_gpu() average < 1.0 ms/call.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402


def _cpu_reference_scores(tel: dict) -> np.ndarray:
    last_pg = np.asarray(tel["last_predicted_gain"], dtype=np.float32)
    pg = np.asarray(tel["predicted_gain"], dtype=np.float32)
    abs_g = np.asarray(tel["abs_proj_grad"], dtype=np.float32)
    base_gain = np.where(last_pg != 0.0, last_pg, pg)
    score = base_gain + np.float32(0.25) * abs_g
    return np.where(np.isfinite(score), score, np.float32(0.0))


def test_ago2_gpu_select_correctness_and_bit_identical_exclusion(tmp_path):
    """GPU active-set selection matches CPU reference and leaves excluded splats bit-exact."""
    ds = sc.build_dataset(tmp_path / "ds_ago2_corr", n_points=500, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_ago2_corr", mcmc_protect_structural_groups=True)
    mod.force_backward_variant(1)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()

    # Warm up 4 steps so Gaussians accumulate non-zero telemetry and update_count >= 2
    for s in range(1, 5):
        mod.train_step(s % int(mod.num_train), s)

    n = int(len(mod.xyz_ws))
    assert n == 500

    # Mark first 50 splats as CONTRADICTED (frozen)
    states = np.full((n,), 5, dtype=np.uint32)  # MEASURED_FREE
    states[:50] = 4  # CONTRADICTED
    mod.set_splat_states(states)

    tel = mod.get_ago_telemetry()
    scores = _cpu_reference_scores(tel)
    eligible_mask = states != 4
    pos_scores = scores[eligible_mask & (scores > 0.0)]
    assert pos_scores.size > 20, f"Expected >20 positive-gain splats, got {pos_scores.size}"

    # Threshold at 60th percentile of positive scores to force a strict non-empty subset
    thr = float(np.quantile(pos_scores, 0.60))
    assert thr > 0.0

    # CPU reference with newborn_warmup_steps=0.0 and exclude_frozen=True
    ref_mask = eligible_mask & (scores >= np.float32(thr))
    ref_indices = set(np.flatnonzero(ref_mask).tolist())
    assert 0 < len(ref_indices) < n

    res = mod.select_ago_active_set_gpu(
        gain_threshold=thr,
        newborn_warmup_steps=0.0,
        exclude_frozen=True,
    )
    gpu_indices = set(np.asarray(mod.get_active_set(), dtype=np.uint32).tolist())

    assert res["num_gaussians"] == n
    assert res["eligible_count"] == int(np.sum(eligible_mask))
    assert res["active_count"] == len(gpu_indices)
    assert 0 < res["active_count"] < n
    assert res["fell_back_to_full"] is False
    assert abs(res["active_fraction"] - (len(gpu_indices) / n)) < 1e-6
    assert gpu_indices == ref_indices, (
        f"GPU vs CPU index mismatch: gpu_only={len(gpu_indices - ref_indices)}, "
        f"cpu_only={len(ref_indices - gpu_indices)}"
    )

    ref_total_gain = float(np.sum(np.maximum(scores[eligible_mask], 0.0)))
    ref_mean_gain = ref_total_gain / float(np.sum(eligible_mask))
    assert abs(res["total_predicted_gain"] - ref_total_gain) <= max(1e-4, 1e-3 * ref_total_gain)
    assert abs(res["mean_predicted_gain"] - ref_mean_gain) <= max(1e-6, 1e-3 * ref_mean_gain)

    # Run a training step and verify excluded Gaussians are bit-identical
    xyz_before = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    sh_before = np.asarray(mod.sh_coeffs, dtype=np.float32).copy()
    rot_before = np.asarray(mod.rotations, dtype=np.float32).copy()
    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).copy()

    mod.train_step(1, 5)

    xyz_after = np.asarray(mod.xyz_ws, dtype=np.float32)
    sh_after = np.asarray(mod.sh_coeffs, dtype=np.float32)
    rot_after = np.asarray(mod.rotations, dtype=np.float32)
    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32)

    excl_idx = np.array(sorted(set(range(n)) - gpu_indices), dtype=np.int64)
    act_idx = np.array(sorted(gpu_indices), dtype=np.int64)
    assert excl_idx.size > 0 and act_idx.size > 0

    assert float(np.max(np.abs(xyz_after[excl_idx] - xyz_before[excl_idx]))) == 0.0
    assert float(np.max(np.abs(sh_after[excl_idx] - sh_before[excl_idx]))) == 0.0
    assert float(np.max(np.abs(rot_after[excl_idx] - rot_before[excl_idx]))) == 0.0
    assert float(np.max(np.abs(so_after[excl_idx] - so_before[excl_idx]))) == 0.0

    # At least some selected Gaussians must have updated
    assert float(np.max(np.abs(xyz_after[act_idx] - xyz_before[act_idx]))) > 0.0

    # Empty-selection fallback: huge threshold + no warmup -> fell_back_to_full=True, training still runs
    res_empty = mod.select_ago_active_set_gpu(
        gain_threshold=1e30,
        newborn_warmup_steps=0.0,
        exclude_frozen=True,
    )
    assert res_empty["active_count"] == 0
    assert res_empty["fell_back_to_full"] is True
    xyz_pre_fb = np.asarray(mod.xyz_ws, dtype=np.float32).copy()
    mod.train_step(2 % int(mod.num_train), 6)
    xyz_post_fb = np.asarray(mod.xyz_ws, dtype=np.float32)
    assert float(np.max(np.abs(xyz_post_fb - xyz_pre_fb))) > 0.0

    mod.cleanup()


def test_ago2_gpu_select_newborn_protection(tmp_path):
    """Gaussians with update_count < newborn_warmup_steps are always included even at huge threshold."""
    ds = sc.build_dataset(tmp_path / "ds_ago2_nb", n_points=300, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_ago2_nb")
    mod.force_backward_variant(1)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()

    for s in range(1, 5):
        mod.train_step(s % int(mod.num_train), s)

    # Append 40 newborn Gaussians (update_count == 0)
    rng = np.random.default_rng(77)
    n_old = int(len(mod.xyz_ws))
    n_new = 40
    xyz, sh, rot, scales, opac, gids_add = sc.make_batch(rng, n_new, -1)
    mod.append_splats_preserving_optimizer(xyz, sh, rot, scales, opac, gids_add)
    n_tot = int(len(mod.xyz_ws))
    assert n_tot == n_old + n_new

    # Mark 10 of the newborns as CONTRADICTED (frozen) to ensure exclude_frozen still wins
    states = np.full((n_tot,), 5, dtype=np.uint32)
    states[n_old : n_old + 10] = 4
    mod.set_splat_states(states)

    res = mod.select_ago_active_set_gpu(
        gain_threshold=1e20,
        newborn_warmup_steps=2.0,
        exclude_frozen=True,
    )
    selected = set(np.asarray(mod.get_active_set(), dtype=np.uint32).tolist())
    assert res["active_count"] == len(selected)

    # The 30 non-frozen newborns MUST be included; the 10 frozen newborns MUST be excluded
    for i in range(n_old, n_old + 10):
        assert i not in selected, f"Frozen newborn {i} should be excluded"
    for i in range(n_old + 10, n_tot):
        assert i in selected, f"Active newborn {i} must be protected by warmup"

    mod.cleanup()


def test_ago2_adaptive_selection_and_step_gain_summary(tmp_path):
    """Adaptive selection + get_ago_step_gain_summary() return finite, consistent scalars."""
    ds = sc.build_dataset(tmp_path / "ds_ago2_adap", n_points=400, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_ago2_adap")
    mod.force_backward_variant(1)
    mod.set_ago_telemetry_enabled(True)
    mod.reset_ago_telemetry()

    for s in range(1, 5):
        mod.train_step(s % int(mod.num_train), s)

    # First call: prev_mean_gain == 0 -> threshold == 0, warms prev_mean_gain_
    r1 = mod.select_ago_adaptive_active_set_gpu(
        target_gain_ratio=0.5,
        newborn_warmup_steps=0.0,
        exclude_frozen=True,
    )
    assert np.isfinite(r1["total_predicted_gain"]) and r1["total_predicted_gain"] > 0.0
    assert np.isfinite(r1["mean_predicted_gain"]) and r1["mean_predicted_gain"] > 0.0
    assert r1["gain_threshold"] == 0.0

    # Second call: uses prev_mean_gain * 0.5 as threshold -> strictly selects high-gain subset
    r2 = mod.select_ago_adaptive_active_set_gpu(
        target_gain_ratio=0.5,
        newborn_warmup_steps=0.0,
        exclude_frozen=True,
    )
    expected_thr = r1["mean_predicted_gain"] * 0.5
    assert abs(r2["gain_threshold"] - expected_thr) <= max(1e-7, 1e-4 * expected_thr)
    assert 0 < r2["active_count"] < r2["num_gaussians"]
    assert 0.0 < r2["active_fraction"] < 1.0

    # Cached 16-byte summary must match r2 without any per-Gaussian readback
    t0 = time.perf_counter()
    s_cached = mod.get_ago_step_gain_summary()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert s_cached["active_count"] == r2["active_count"]
    assert s_cached["num_gaussians"] == r2["num_gaussians"]
    assert abs(s_cached["active_fraction"] - r2["active_fraction"]) < 1e-9
    assert abs(s_cached["total_predicted_gain"] - r2["total_predicted_gain"]) < 1e-9
    assert abs(s_cached["mean_predicted_gain"] - r2["mean_predicted_gain"]) < 1e-9
    assert dt_ms < 1.0, f"get_ago_step_gain_summary took {dt_ms:.3f} ms"

    mod.cleanup()


def test_ago2_gpu_select_latency_benchmark(tmp_path):
    """20 calls to select_ago_adaptive_active_set_gpu average < 1.0 ms/call on 50k splats."""
    ds = sc.build_dataset(tmp_path / "ds_ago2_lat", n_points=500, w=64, h=48, structured=True)
    mod = sc.new_trainer(ds, tmp_path / "out_ago2_lat")
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

    # Warmup 3 calls
    for _ in range(3):
        mod.select_ago_adaptive_active_set_gpu(0.5, 2.0, True)

    # Benchmark 20 GPU adaptive selection calls
    n_iters = 20
    t0 = time.perf_counter()
    last_res = None
    for _ in range(n_iters):
        last_res = mod.select_ago_adaptive_active_set_gpu(0.5, 2.0, True)
    gpu_avg_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    # Compare against legacy CPU round-trip (get_ago_telemetry + set_active_set)
    t1 = time.perf_counter()
    for _ in range(5):
        tel = mod.get_ago_telemetry()
        scores = _cpu_reference_scores(tel)
        idx = np.flatnonzero(scores >= float(last_res["gain_threshold"]))
        mod.set_active_set(idx)
    cpu_avg_ms = (time.perf_counter() - t1) * 1000.0 / 5

    print(
        f"[AGO-2 GPU Select Benchmark N={n_tot}] "
        f"gpu_adaptive={gpu_avg_ms:.3f} ms/call vs cpu_roundtrip={cpu_avg_ms:.3f} ms/call "
        f"({cpu_avg_ms / max(gpu_avg_ms, 1e-6):.1f}x speedup, active={last_res['active_count']}/{n_tot})"
    )
    assert gpu_avg_ms < 1.0, f"Expected < 1.0 ms/call, got {gpu_avg_ms:.3f} ms/call"

    mod.cleanup()
