"""Frozen-splat backward skip, update_group_sparse and compact_retracted (all GPU-side, no full readback).

Scenario: groups 10 (normal), 11 (IMAGINED), 12 (protected LOCKED), 13 (retracted for compaction) plus the
ungrouped initial points. mcmc_protect_structural_groups=True so that no Morton reordering moves splats.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "vksplat" / "src"
IMAGINED, LOCKED = 11, 12
GRADS = ("v_xy_vs", "v_inv_cov_vs_opacity", "v_rgb")
R = 32  # SH_REORDER_SIZE


def _build(tmp_path, variant=1, groups=(10, 11, 12, 13), n_points=300, size=(64, 48)):
    ds = sc.build_dataset(tmp_path / "ds", n_points=n_points, w=size[0], h=size[1])
    mod = sc.new_trainer(ds, tmp_path / "out", mcmc_protect_structural_groups=True)
    mod.force_backward_variant(variant)
    rng = np.random.default_rng(3)
    for gid in groups:
        mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 48, gid))
    return mod


def _train(mod, steps, first=1):
    n = int(mod.num_train)
    for s in range(first, first + steps):
        mod.train_step(s % n, s)


def _gradients(tmp_path, variant, skip, tag):
    """Raster-backward gradients of the first training step (the buffers keep them after the step)."""
    mod = _build(tmp_path / tag, variant)
    mod.set_freeze_skip_backward(skip)
    mod.set_group_states(np.array([IMAGINED, LOCKED], dtype=np.int32), np.array([1, 2], dtype=np.uint32))
    gids = sc.snapshot(mod)["gids"]
    _train(mod, 1)
    shapes = {"v_xy_vs": 2, "v_inv_cov_vs_opacity": 4, "v_rgb": 3}
    return gids, {k: np.asarray(getattr(mod, k), dtype=np.float32).reshape(-1, shapes[k])[:len(gids)].copy() for k in GRADS}


@pytest.mark.parametrize("variant", [1, 3])
def test_frozen_splats_skip_backward_accumulation(tmp_path, variant):
    gids, ref_a = _gradients(tmp_path, variant, False, "a")
    _, ref_b = _gradients(tmp_path, variant, False, "b")
    _, on = _gradients(tmp_path, variant, True, "c")
    imagined, locked = gids == IMAGINED, gids == LOCKED
    other = ~(imagined | locked)
    for k in GRADS:
        noise = np.abs(ref_a[k] - ref_b[k]).max()
        tol = 4 * noise + 1e-5 * np.abs(ref_a[k]).max()
        assert np.abs(on[k][other] - ref_a[k][other]).max() <= tol, k
    # the reference really has gradients on the frozen splats, the skipping run has none (exact zeros)
    for k in ("v_xy_vs", "v_inv_cov_vs_opacity"):
        assert np.abs(ref_a[k][imagined]).max() > 0 and np.abs(ref_a[k][locked]).max() > 0, k
        assert np.all(on[k][imagined] == 0) and np.all(on[k][locked] == 0), k
    assert np.all(on["v_rgb"][locked] == 0) and np.abs(ref_a["v_rgb"][locked]).max() > 0
    noise = np.abs(ref_a["v_rgb"] - ref_b["v_rgb"]).max()
    assert np.abs(on["v_rgb"][imagined] - ref_a["v_rgb"][imagined]).max() <= 4 * noise + 1e-5 * np.abs(ref_a["v_rgb"]).max()


def test_skipping_training_matches_reference_within_noise(tmp_path):
    def run(skip, sub):
        mod = _build(tmp_path / sub)
        mod.set_freeze_skip_backward(skip)
        mod.set_group_states(np.array([IMAGINED, LOCKED], dtype=np.int32), np.array([1, 2], dtype=np.uint32))
        _train(mod, 12)
        return sc.snapshot(mod)

    ref_a, ref_b, on = run(False, "a"), run(False, "b"), run(True, "c")
    gids = on["gids"]
    frozen = np.isin(gids, (IMAGINED, LOCKED))
    for k in ("xyz", "rot", "scales", "opac", "sh"):
        noise = np.abs(ref_a[k] - ref_b[k]).max()
        assert np.abs(on[k][~frozen] - ref_a[k][~frozen]).max() <= 4 * noise + 1e-6, k
    for k in ("xyz", "rot", "scales", "opac"):  # frozen geometry is bit-identical with and without the skip
        assert np.array_equal(on[k][frozen], ref_a[k][frozen]), k
    imagined = gids == IMAGINED
    noise = np.abs(ref_a["sh"] - ref_b["sh"]).max()
    assert np.abs(on["sh"][imagined] - ref_a["sh"][imagined]).max() <= 4 * noise + 1e-6
    assert np.array_equal(on["sh"][gids == LOCKED], ref_a["sh"][gids == LOCKED])


def _row(arr, i, words):
    return arr.reshape(-1, words)[i]


def _sh_row(arr, i):
    return arr.reshape(-1, 4)[[12 * (i // R) * R + d * R + i % R for d in range(12)]]


def _state(mod):
    snap = mod.snapshot_optimizer_state()
    snap.update(mod.get_optimizer_moments())
    return int(snap["num_splats"]), snap


def test_update_group_sparse_rewrites_only_that_group(tmp_path):
    mod = _build(tmp_path)
    _train(mod, 6)
    n, before = _state(mod)
    gids = np.asarray(before["splat_group_ids"])
    rows = np.where(gids == IMAGINED)[0]
    rng = np.random.default_rng(9)
    xyz = rng.uniform(-0.5, 0.5, size=(len(rows), 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.03, size=(len(rows), 3)).astype(np.float32)
    quats = rng.normal(size=(len(rows), 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    mod.update_group_sparse(IMAGINED, xyz, scales, quats)
    _, after = _state(mod)

    mine = np.zeros(n, bool)
    mine[rows] = True
    xs = np.asarray(after["xyz_ws"]).reshape(-1, 3)
    assert np.array_equal(xs[rows], xyz)
    assert np.array_equal(np.asarray(after["rotations"]).reshape(-1, 4)[rows], quats)
    assert np.array_equal(np.asarray(after["scales_opacs"]).reshape(-1, 4)[rows, :3], scales)
    # everything else, and the group's opacity / SH / moments of opacity+SH, is untouched
    for key, words in (("xyz_ws", 3), ("rotations", 4), ("scales_opacs", 4)):
        a, b = np.asarray(before[key]).reshape(-1, words)[:n], np.asarray(after[key]).reshape(-1, words)[:n]
        assert np.array_equal(a[~mine], b[~mine]), key
    assert np.array_equal(np.asarray(before["scales_opacs"]).reshape(-1, 4)[rows, 3],
                          np.asarray(after["scales_opacs"]).reshape(-1, 4)[rows, 3])
    assert np.array_equal(before["sh_coeffs"], after["sh_coeffs"])
    for key in ("g_sh_coeffs_1", "g_sh_coeffs_2"):
        assert np.array_equal(before[key], after[key]), key
    # geometry moments of the group are reset (and were non-zero), the rest keeps them
    for key, words in (("g_xyz_ws", 6), ("g_rotations", 8), ("g_scales_opacs", 8)):
        a, b = np.asarray(before[key]).reshape(-1, words)[:n], np.asarray(after[key]).reshape(-1, words)[:n]
        assert np.array_equal(a[~mine], b[~mine]), key
        if key == "g_scales_opacs":
            assert np.all(b[rows][:, [0, 1, 2, 4, 5, 6]] == 0) and np.array_equal(b[rows][:, [3, 7]], a[rows][:, [3, 7]])
        else:
            assert np.abs(a[rows]).max() > 0 and np.all(b[rows] == 0), key
    with pytest.raises(ValueError):
        mod.update_group_sparse(IMAGINED, xyz[:-1], scales[:-1], quats[:-1])
    _train(mod, 3, first=7)
    assert np.isfinite(sc.snapshot(mod)["xyz"]).all()


def test_compact_retracted_preserves_live_splats_exactly(tmp_path):
    mod = _build(tmp_path, groups=(10, 13, 11, 14, 12))
    mod.set_group_states(np.array([IMAGINED, LOCKED], dtype=np.int32), np.array([1, 2], dtype=np.uint32))
    _train(mod, 8)
    n0, before = _state(mod)
    state_before = np.asarray(mod.get_device_group_state()).copy()
    mod.retract_group_sparse(13)
    mod.retract_group_sparse(14)
    n0, before = _state(mod)  # retraction zeroes opacity of the retracted groups
    gids0 = np.asarray(before["splat_group_ids"])[:n0]
    index_map = np.asarray(mod.compact_retracted())
    n1, after = _state(mod)

    dead = np.isin(gids0, (13, 14))
    assert index_map.shape == (n0,) and n1 == int((~dead).sum())
    assert np.all(index_map[dead] == -1)
    assert np.array_equal(index_map[~dead], np.arange(n1))  # order preserved
    live = np.where(~dead)[0]
    gids1 = np.asarray(after["splat_group_ids"])[:n1]
    assert np.array_equal(gids1, gids0[live])
    assert np.array_equal(np.asarray(mod.get_device_group_ids())[:n1], gids1)
    assert not np.isin(gids1, (13, 14)).any()
    # group states survive (retracted groups stay RETRACTED, nothing references them any more)
    state_after = np.asarray(mod.get_device_group_state())
    assert state_after[IMAGINED] == 1 and state_after[LOCKED] == 2 and state_after[13] == 3 and state_after[14] == 3
    assert np.array_equal(state_after[:10], state_before[:10])

    for key, words in (("xyz_ws", 3), ("rotations", 4), ("scales_opacs", 4),
                       ("g_xyz_ws", 6), ("g_rotations", 8), ("g_scales_opacs", 8)):
        a = np.asarray(before[key]).reshape(-1, words)[:n0][live]
        b = np.asarray(after[key]).reshape(-1, words)[:n1]
        assert np.array_equal(a, b), key
    for key in ("sh_coeffs", "g_sh_coeffs_1", "g_sh_coeffs_2"):
        if key == "sh_coeffs":
            a = np.asarray(before[key]).reshape(n0, -1)[live]
            b = np.asarray(after[key]).reshape(n1, -1)
        else:
            a = np.stack([_sh_row(np.asarray(before[key]), i) for i in live])
            b = np.stack([_sh_row(np.asarray(after[key]), i) for i in range(n1)])
        assert np.array_equal(a, b), key
    # padding lanes of the last SH block are zero
    pad = np.asarray(after["g_sh_coeffs_1"]).reshape(-1, 4)
    lanes = [12 * (n1 // R) * R + d * R + l for d in range(12) for l in range(n1 % R, R)] if n1 % R else []
    assert all(np.all(pad[j] == 0) for j in lanes)

    # the compacted map keeps training and appending
    _train(mod, 4, first=9)
    mod.append_splats_preserving_optimizer(*sc.make_batch(np.random.default_rng(1), 20, 15))
    _train(mod, 2, first=13)
    snap = sc.snapshot(mod)
    assert len(snap["gids"]) == n1 + 20 and np.isfinite(snap["xyz"]).all()


def test_compact_without_retracted_groups_is_identity(tmp_path):
    mod = _build(tmp_path, groups=(10, 11))
    n = len(sc.snapshot(mod)["gids"])
    before = sc.snapshot(mod)
    assert np.array_equal(np.asarray(mod.compact_retracted()), np.arange(n))
    after = sc.snapshot(mod)
    for k in before:
        assert np.array_equal(before[k], after[k]), k


def _body(text, signature):
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    for j in range(i, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            return text[start:j + 1]
    raise AssertionError(signature)


def test_group_ops_have_no_readback():
    trainer = (SRC / "gs_trainer.cpp").read_text()
    bindings = (SRC / "python_bindings.cpp").read_text()
    forbidden = re.compile(r"copyFromDevice\(|downloadDeviceSlice|buffer_to_array|vkWaitForFences|HOST_GUARD|vkQueueWaitIdle|readElement")
    for text, sig in (
        (trainer, "void VulkanGSTrainer::update_group_sparse("),
        (trainer, "std::vector<int32_t> VulkanGSTrainer::compact_retracted("),
        (bindings, "void update_group_sparse("),
        (bindings, "py::array_t<int32_t> compact_retracted("),
    ):
        assert not forbidden.search(_body(text, sig)), sig
