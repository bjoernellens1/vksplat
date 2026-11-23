"""Unit tests for native GPU-resident sparse opacity decay (`decay_opacities_sparse`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402


def _cpu_logit_decay(opac: np.ndarray, delta_logit: float, min_opacity: float) -> np.ndarray:
    p = np.clip(opac.astype(np.float32), 1e-7, 1.0 - 1e-7)
    logit = np.log(p / (1.0 - p)) - np.float32(delta_logit)
    new_p = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
    return np.maximum(new_p, np.float32(min_opacity)).astype(np.float32)


def test_decay_opacities_sparse_parity_and_moments(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=320, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    n_train = int(mod.num_train)
    for s in range(1, 12):
        mod.train_step(s % n_train, s)

    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4).copy()
    mom_before = np.asarray(mod.get_optimizer_moments()["g_scales_opacs"], dtype=np.float32).reshape(-1, 2, 4).copy()
    n = len(so_before)
    assert n >= 100

    rows = np.array([0, 5, 17, 63, 64, 65, n - 1], dtype=np.uint32)
    mask = np.zeros((n,), dtype=bool)
    mask[rows] = True

    delta_logit = 3.0
    min_opacity = 0.002
    mod.decay_opacities_sparse(rows, delta_logit, min_opacity, True, True)

    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
    mom_after = np.asarray(mod.get_optimizer_moments()["g_scales_opacs"], dtype=np.float32).reshape(-1, 2, 4)

    expected_op = _cpu_logit_decay(so_before[rows, 3], delta_logit, min_opacity)
    assert float(np.max(np.abs(so_after[rows, 3] - expected_op))) <= 1e-6
    assert np.array_equal(so_after[rows, :3], so_before[rows, :3])
    assert np.array_equal(so_after[~mask], so_before[~mask])
    assert np.array_equal(mom_after[~mask], mom_before[~mask])
    assert np.array_equal(mom_after[rows, :, :3], mom_before[rows, :, :3])
    assert np.all(mom_after[rows, :, 3] == 0.0)


def test_decay_opacities_sparse_dedup_validation_and_bounds(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=128, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    mod.train_step(0, 1)

    # Parameter validation (even on empty rows)
    for bad_delta in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            mod.decay_opacities_sparse(np.array([], dtype=np.uint32), bad_delta, 1e-4, True)
    for bad_min in (0.0, -0.1, 1.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            mod.decay_opacities_sparse(np.array([], dtype=np.uint32), 2.5, bad_min, True)

    # Near-zero, typical (0.8 -> ~0.247), and near-one inputs + duplicate rows
    so0 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4).copy()
    so0[0, 3] = 1e-6
    so0[1, 3] = 0.80
    so0[2, 3] = 0.999999
    mod._set_scales_opacs(np.ascontiguousarray(so0, dtype=np.float32))

    # Duplicate row indices must apply decay exactly once (no GPU race / double decay)
    dup_rows = np.array([1, 0, 1, 2, 1, 0, 2, 2], dtype=np.uint32)
    mod.decay_opacities_sparse(dup_rows, 2.5, 1e-4, False, True)
    so1 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
    expected = _cpu_logit_decay(np.array([1e-6, 0.80, 0.999999], dtype=np.float32), 2.5, 1e-4)
    assert float(np.max(np.abs(so1[:3, 3] - expected))) <= 1e-6
    assert abs(float(so1[1, 3]) - 0.24718) < 1e-3
    assert np.array_equal(so1[3:], so0[3:])


def test_decay_opacities_sparse_protected_states_and_groups(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=128, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    mod.train_step(0, 1)

    n = int(np.asarray(mod.xyz_ws).shape[0])
    # Set splat states: 0=PREDICTED_UNOBSERVED, 1=OBSERVED_UNCERTAIN, 2=OBSERVED_TRAINABLE, 3=METRIC_VERIFIED, 5=FREE
    states = np.full((n,), 5, dtype=np.uint32)
    states[0] = 0  # protected
    states[1] = 1  # protected
    states[2] = 2  # eligible
    states[3] = 3  # protected (METRIC_VERIFIED)
    mod.set_splat_states(states)

    # Put splat 4 in IMAGINED group 10 and splat 5 in LOCKED group 11
    gids = np.full((n,), -1, dtype=np.int32)
    gids[4] = 10
    gids[5] = 11
    mod.set_splat_group_ids(gids)
    mod.set_group_states(np.array([10, 11], dtype=np.int32), np.array([1, 2], dtype=np.uint32))

    so_before = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4).copy()
    rows = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.uint32)
    mod.decay_opacities_sparse(rows, 3.0, 1e-4, True, True)
    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)

    # Protected rows (0, 1, 3, 4, 5) must remain bit-identical
    for idx in (0, 1, 3, 4, 5):
        assert so_after[idx, 3] == so_before[idx, 3], f"Protected row {idx} was decayed!"
    # Eligible rows (2, 6) must be decayed
    for idx in (2, 6):
        assert so_after[idx, 3] < so_before[idx, 3] - 1e-3, f"Eligible row {idx} was not decayed!"

    # Explicit override respect_protected_states=False decays protected rows too
    st_override = mod.decay_opacities_sparse(np.array([0, 3], dtype=np.uint32), 3.0, 1e-4, True, False)
    assert st_override == {"proposed": 2, "deduped": 2, "skipped_out_of_range": 0, "skipped_protected": 0, "applied": 2}
    so_override = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
    assert so_override[0, 3] < so_before[0, 3] - 1e-3
    assert so_override[3, 3] < so_before[3, 3] - 1e-3


def test_splat_uids_lifecycle_and_decay_stats(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=128, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    uids0 = np.asarray(mod.get_splat_uids(), dtype=np.uint64)
    assert len(uids0) == 128
    assert len(np.unique(uids0)) == 128

    # Append preserves existing UIDs and assigns strictly larger monotonic UIDs
    rng = np.random.default_rng(7)
    mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 24, 10))
    mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 16, 11))
    uids1 = np.asarray(mod.get_splat_uids(), dtype=np.uint64)
    assert len(uids1) == 168
    assert np.array_equal(uids1[:128], uids0)
    assert np.all(uids1[128:] > uids0.max())
    assert len(np.unique(uids1)) == 168

    # Compact retracted group 10 preserves exact UIDs of surviving splats (0..127 and group 11)
    mod.set_group_state(10, 3)  # RETRACTED
    mod.compact_retracted()
    uids2 = np.asarray(mod.get_splat_uids(), dtype=np.uint64)
    assert len(uids2) == 144
    assert np.array_equal(uids2[:128], uids0)
    assert np.array_equal(uids2[128:], uids1[152:])

    # Snapshot / restore round-trip preserves splat_uids bit-identically
    snap = mod.snapshot_optimizer_state()
    mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 8, 12))
    assert len(mod.get_splat_uids()) == 152
    mod.restore_optimizer_state(snap)
    assert np.array_equal(np.asarray(mod.get_splat_uids(), dtype=np.uint64), uids2)

    # Exact DecayOpacityStats accounting (duplicates + OOB + protected + eligible)
    states = np.full((len(uids2),), 5, dtype=np.uint32)
    states[0] = 3  # METRIC_VERIFIED (protected)
    mod.set_splat_states(states)
    st = mod.decay_opacities_sparse(np.array([0, 0, 2, 2, 3, 99999], dtype=np.uint32), 2.5, 1e-4, True, True)
    assert st["proposed"] == 6
    assert st["skipped_out_of_range"] == 1
    assert st["deduped"] == 3
    assert st["skipped_protected"] == 1
    assert st["applied"] == 2
