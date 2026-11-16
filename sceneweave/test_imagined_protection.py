"""IMAGINED (state 1) groups: geometry/opacity are frozen on the GPU, SH still trains, other splats move."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

IMAGINED_GID = 11
GEOM = ("xyz", "rot", "scales", "opac")


def _build(tmp_path, **overrides):
    ds = sc.build_dataset(tmp_path / "ds", n_points=400)
    mod = sc.new_trainer(ds, tmp_path / "out", **overrides)
    rng = np.random.default_rng(3)
    for gid in (10, IMAGINED_GID, 12):
        mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 48, gid))
    return mod


def _group_rows(snap, gid):
    """Rows of one group, sorted so that Morton reordering does not matter."""
    m = snap["gids"] == gid
    rows = np.concatenate([snap[k][m].reshape(int(m.sum()), -1) for k in GEOM], axis=1)
    return rows[np.lexsort(rows.T[::-1])]


def _train(mod, steps):
    n_train = int(mod.num_train)
    for s in range(1, steps + 1):
        mod.train_step(s % n_train, s)


def test_capability_flag(tmp_path):
    mod = _build(tmp_path)
    assert mod.supports_imagined_protection() is True
    assert mod.get_mcmc_stats()["imagined_protection"] is True


def test_imagined_group_is_frozen_but_sh_trains(tmp_path):
    mod = _build(tmp_path)
    assert not mod.get_mcmc_stats()["mcmc_protect_structural_groups"]
    mod.set_group_states(np.array([IMAGINED_GID], dtype=np.int32), np.array([1], dtype=np.uint32))
    before = sc.snapshot(mod)
    _train(mod, 12)
    after = sc.snapshot(mod)
    assert np.array_equal(before["gids"], after["gids"])
    m = before["gids"] == IMAGINED_GID
    for k in GEOM:
        assert np.array_equal(before[k][m], after[k][m]), k
    assert np.abs(before["sh"][m] - after["sh"][m]).max() > 0
    assert np.abs(before["xyz"][~m] - after["xyz"][~m]).max() > 0


def test_imagined_group_survives_mcmc_refinement(tmp_path):
    mod = _build(tmp_path, refine_start_iter=0, refine_stop_iter=1000, refine_every=2, strategy="mcmc", cap_max=2000, noise_lr=5e5)
    mod.set_group_state(IMAGINED_GID, 1)
    before = _group_rows(sc.snapshot(mod), IMAGINED_GID)
    _train(mod, 16)
    stats = mod.get_mcmc_stats()
    assert stats["relocated_count"] + stats["added_count"] > 0, "refinement did not run"
    after = _group_rows(sc.snapshot(mod), IMAGINED_GID)
    assert np.array_equal(before, after)


def test_state_transitions_release_protection(tmp_path):
    mod = _build(tmp_path)
    mod.set_group_state(IMAGINED_GID, 1)
    _train(mod, 3)
    mid = sc.snapshot(mod)
    mod.set_group_state(IMAGINED_GID, 0)
    _train(mod, 6)
    end = sc.snapshot(mod)
    m = mid["gids"] == IMAGINED_GID
    assert np.abs(mid["xyz"][m] - end["xyz"][m]).max() > 0


def test_batched_setter_equals_single(tmp_path):
    mod = _build(tmp_path)
    mod.set_group_states(np.array([12, 10, 14], dtype=np.int32), np.array([1, 2, 1], dtype=np.uint32))
    state = np.asarray(mod.get_device_group_state())
    assert list(state[[10, 11, 12, 13, 14]]) == [2, 0, 1, 0, 1]
