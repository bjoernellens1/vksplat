"""Unit tests for native GPU-resident sparse opacity decay (`decay_opacities_sparse`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    mod.decay_opacities_sparse(rows, delta_logit, min_opacity, True)

    so_after = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4)
    mom_after = np.asarray(mod.get_optimizer_moments()["g_scales_opacs"], dtype=np.float32).reshape(-1, 2, 4)

    expected_op = _cpu_logit_decay(so_before[rows, 3], delta_logit, min_opacity)
    assert float(np.max(np.abs(so_after[rows, 3] - expected_op))) <= 1e-6
    assert np.array_equal(so_after[rows, :3], so_before[rows, :3])
    assert np.array_equal(so_after[~mask], so_before[~mask])
    assert np.array_equal(mom_after[~mask], mom_before[~mask])
    assert np.array_equal(mom_after[rows, :, :3], mom_before[rows, :, :3])
    assert np.all(mom_after[rows, :, 3] == 0.0)


def test_decay_opacities_sparse_edge_cases(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=128, w=64, h=48)
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    mod.train_step(0, 1)

    so0 = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4).copy()
    mom0 = np.asarray(mod.get_optimizer_moments()["g_scales_opacs"], dtype=np.float32).reshape(-1, 2, 4).copy()
    n = len(so0)

    # Empty & out-of-bounds rows must be safe no-ops
    mod.decay_opacities_sparse(np.array([], dtype=np.uint32), 2.5, 1e-4, True)
    mod.decay_opacities_sparse(np.array([n, n + 100], dtype=np.uint32), 2.5, 1e-4, True)
    assert np.array_equal(np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4), so0)

    # reset_moments=False must preserve Adam moments on affected rows too
    rows = np.array([1, 2, 3], dtype=np.uint32)
    mod.decay_opacities_sparse(rows, 1.5, 1e-4, False)
    mom1 = np.asarray(mod.get_optimizer_moments()["g_scales_opacs"], dtype=np.float32).reshape(-1, 2, 4)
    assert np.array_equal(mom1, mom0)
