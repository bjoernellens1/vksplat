"""Sparse tile backward: per-tile residual -> quantile threshold -> halo dilation -> compacted tile list ->
indirect dispatch of the raster and SSIM backward kernels (sparse_tiles.slang).

Sparse objective: on a sparse step only the active tiles contribute gradient. That is exactly the dense L1 loss
restricted to the active tiles (SSIM windows read full-frame statistics, so the SSIM term is the full-frame SSIM
gradient evaluated on the active pixels); inactive tiles contribute zero. Every `sparse_full_every`-th step is dense.

The backward pass uses float atomics, so identical runs differ by run-to-run noise; comparisons are bounded by 4x
that noise (plus a small floor), as in test_train_steps.py.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402
import gen_reference_snapshot as ref  # noqa: E402
from test_train_steps import FORBIDDEN, SRC, _body, _loss, _noise  # noqa: E402

FLOOR = 1e-4
TILE = 16
# loss regression (sparse vs dense, converged scene, 100 more steps) accepted by the test, in percent
CONVERGED_LOSS_REGRESSION_PCT = 5.0


@pytest.fixture(scope="module")
def dataset():
    with tempfile.TemporaryDirectory() as td:
        yield sc.build_dataset(Path(td) / "ds", n_images=4, n_points=600, w=128, h=96)


@pytest.fixture(scope="module")
def converged_dataset():
    with tempfile.TemporaryDirectory() as td:
        yield sc.build_dataset(Path(td) / "ds", n_images=4, n_points=800, w=128, h=96, structured=True)


def _train(ds, out, steps, **cfg):
    mod = sc.new_trainer(ds, out, **cfg)
    n = int(mod.num_train)
    for step in range(steps):
        mod.train_step(step % n, step)
    return mod


def _params(mod):
    return {k: v for k, v in sc.snapshot(mod).items() if k != "gids"}


def _noise_of(runs):
    """Largest pairwise difference per parameter over >= 3 identical runs (two runs underestimate the noise)."""
    pairs = [_noise(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    return {k: max(p[k] for p in pairs) for k in runs[0]}


def _close(a, b, noise, what, floor=FLOOR):
    for k in a:
        tol = max(4 * noise[k], floor)
        diff = float(np.abs(a[k] - b[k]).max())
        assert diff <= tol, f"{what}: {k} differs by {diff:.3e} > {tol:.3e}"


def test_sparse_off_matches_4eda56f(tmp_path):
    """quantile 0 (default): results equal the 4eda56f dense build within 4x run-to-run atomic noise."""
    with tempfile.TemporaryDirectory() as td:
        ds = sc.build_dataset(Path(td) / "ds", n_points=ref.N_POINTS)
        runs = [_params(_train(ds, tmp_path / f"r{i}", ref.STEPS)) for i in range(3)]
    base = dict(np.load(Path(__file__).resolve().parent / "reference_4eda56f.npz"))
    _close(runs[0], base, _noise_of(runs), "vs 4eda56f")


def _expected_active_tiles(mod, slot, quantile, halo):
    """numpy mirror of sparse_tile_residual + sparse_tile_select for the step that trains on `slot`."""
    mod.render_train(slot)
    img = np.asarray(mod.get_train_image(slot), dtype=np.float32)
    h, w = img.shape[:2]
    pred = np.maximum(np.asarray(mod.pixel_state, dtype=np.float32).reshape(h, w, 4)[..., :3], 0.0)
    alpha = img[..., 3:4] / 255.0 if img.shape[-1] == 4 else 1.0
    err = np.abs(pred - img[..., :3] / 255.0) * alpha
    gh, gw = -(-h // TILE), -(-w // TILE)
    resid = np.zeros((gh, gw), np.float32)
    for ty in range(gh):
        for tx in range(gw):
            resid[ty, tx] = err[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE].mean()
    keep = max(1, int(np.ceil((1.0 - quantile) * gh * gw)))
    thresh = np.sort(resid.ravel())[::-1][keep - 1]
    mask = (resid >= thresh) & (resid > 0)
    dil = np.zeros_like(mask)
    for dy in range(-halo, halo + 1):
        for dx in range(-halo, halo + 1):
            sy, sx = np.clip(np.arange(gh) + dy, 0, gh - 1), np.clip(np.arange(gw) + dx, 0, gw - 1)
            valid_y = (np.arange(gh) + dy >= 0) & (np.arange(gh) + dy < gh)
            valid_x = (np.arange(gw) + dx >= 0) & (np.arange(gw) + dx < gw)
            dil |= mask[np.ix_(sy, sx)] & valid_y[:, None] & valid_x[None, :]
    return int(dil.sum()), gh * gw


@pytest.mark.parametrize("halo", [0, 1])
def test_sparse_dispatches_fewer_tiles_and_sentinel_is_full(converged_dataset, tmp_path, halo):
    q, every = 0.6, 4
    mod = _train(converged_dataset, tmp_path / "a", 8, sparse_tile_quantile=0.0)  # let the render settle a bit
    mod.set_sparse_backward(q, halo, every)
    n = int(mod.num_train)
    sparse_steps = 0
    for step in range(8, 8 + 3 * every):
        slot = step % n
        expect, total = _expected_active_tiles(mod, slot, q, halo) if step % every else (None, None)
        before = mod.get_work_counters()
        mod.train_step(slot, step)
        after = mod.get_work_counters()
        dispatched = after["backward_tiles_dispatched"] - before["backward_tiles_dispatched"]
        grid = after["backward_tiles_total"] - before["backward_tiles_total"]
        if step % every == 0:
            assert dispatched == grid, f"sentinel step {step} is not full"
            assert after["backward_sparse_passes"] == before["backward_sparse_passes"]
        else:
            sparse_steps += 1
            assert after["backward_sparse_passes"] - before["backward_sparse_passes"] == 1
            assert dispatched < grid, f"step {step}: sparse step dispatched every tile"
            assert abs(dispatched - expect) <= 3, f"step {step}: GPU selected {dispatched} tiles, numpy {expect}"
    assert sparse_steps == 3 * (every - 1)
    c = mod.get_work_counters()
    assert c["backward_tiles_dispatched"] < c["backward_tiles_total"]


def test_all_tiles_listed_equals_dense(dataset, tmp_path):
    """A quantile so small that every tile is kept runs the list kernels over all tiles: same result as dense."""
    dense = [_params(_train(dataset, tmp_path / f"d{i}", 12)) for i in range(3)]
    mod = _train(dataset, tmp_path / "s", 12, sparse_tile_quantile=1e-6, sparse_tile_halo=1, sparse_full_every=100)
    c = mod.get_work_counters()
    assert c["backward_sparse_passes"] >= 10 and c["backward_tiles_dispatched"] == c["backward_tiles_total"]
    _close(_params(mod), dense[0], _noise_of(dense), "list over all tiles")


def test_converged_scene_loss_does_not_regress(converged_dataset, tmp_path):
    """Train 200 dense steps, then 100 more dense vs 100 more sparse (quantile 0.5): loss regression bound."""
    arms = {}
    for name, sparse in (("dense", False), ("sparse", True), ("dense2", False)):
        mod = _train(converged_dataset, tmp_path / name, 200)
        if sparse:
            mod.set_sparse_backward(0.5, 1, 4)
        n = int(mod.num_train)
        for step in range(200, 300):
            mod.train_step(step % n, step)
        arms[name] = _loss(mod)
        if sparse:
            c = mod.get_work_counters()
            assert c["backward_tiles_dispatched"] < c["backward_tiles_total"]
    regression = 100.0 * (arms["sparse"] - arms["dense"]) / arms["dense"]
    noise = 100.0 * abs(arms["dense2"] - arms["dense"]) / arms["dense"]
    print(f"\nconverged-scene L1: dense {arms['dense']:.5f} sparse {arms['sparse']:.5f} "
          f"regression {regression:+.2f}% (dense run-to-run {noise:.2f}%)")
    assert regression <= max(CONVERGED_LOSS_REGRESSION_PCT, 4 * noise)


def test_sparse_step_path_has_no_readback():
    """The selection and both backward dispatches are recorded GPU-side: no fence, readback or host sync."""
    for fname, func in [("gs_renderer.cpp", "executeSelectSparseTiles"), ("gs_renderer.cpp", "executeRasterizeBackward"),
                        ("gs_trainer.cpp", "executeComputeSSIMGradient"), ("python_bindings.cpp", "begin_step"),
                        ("python_bindings.cpp", "compute_pixel_loss_grad")]:
        body = _body(SRC / fname, func)
        for token in FORBIDDEN:
            assert token not in body, f"{func} contains {token}"
    # the counter readback exists, but only outside the step path
    for fname, func in [("python_bindings.cpp", "train_steps"), ("python_bindings.cpp", "train_step_gated"),
                        ("python_bindings.cpp", "train_step_sync")]:
        assert "sparseTilesDispatched" not in _body(SRC / fname, func), func
    assert "readElement" in _body(SRC / "gs_renderer.cpp", "sparseTilesDispatched")


def test_one_fence_per_batch_with_sparse(dataset, tmp_path):
    mod = sc.new_trainer(dataset, tmp_path / "f", sparse_tile_quantile=0.5, sparse_full_every=2)
    n = int(mod.num_train)
    step = 0
    for _ in range(16):  # warm-up: descriptor updates of every backward variant
        mod.train_steps([step % n], step)
        step += 1
    for batch in (1, 2, 4):
        before = mod.get_work_counters()["fence_waits_total"]
        for _ in range(3):
            mod.train_steps([(step + i) % n for i in range(batch)], step)
            step += batch
        assert mod.get_work_counters()["fence_waits_total"] - before == 3, f"batch={batch}"


def test_sparse_batched_matches_single(dataset, tmp_path):
    """Gated batches record the same sparse steps as single calls (indirect arguments are per step)."""
    kw = dict(sparse_tile_quantile=0.5, sparse_tile_halo=1, sparse_full_every=3)
    singles = [_params(_train(dataset, tmp_path / f"s{i}", 12, **kw)) for i in range(3)]
    mod = sc.new_trainer(dataset, tmp_path / "b", **kw)
    n = int(mod.num_train)
    for step in range(0, 12, 4):
        mod.train_steps([(step + i) % n for i in range(4)], step)
    # a borderline tile can flip between runs (the selection is discontinuous in the float residual): a looser floor
    _close(_params(mod), singles[0], _noise_of(singles), "batched sparse", floor=2e-3)


def test_gate_overflow_skips_sparse_steps_on_gpu(dataset, tmp_path):
    """A step skipped by the step gate dispatches no tiles (its indirect arguments are zeroed on the GPU), so the
    GPU tile counter of an overflowing batch equals that of the same steps without overflow."""
    kw = dict(sparse_tile_quantile=0.5, sparse_tile_halo=1, sparse_full_every=100)
    dispatched = {}
    for force in (False, True):
        mod = sc.new_trainer(dataset, tmp_path / f"o{force}", **kw)
        mod.train_step(0, 0)  # refine step: synchronous path, sizes the buffers
        before = mod.get_work_counters()["backward_tiles_dispatched"]
        if force:
            mod._force_isect_capacity(16)
        mod.train_steps([1, 2, 0], 1)
        c = mod.get_work_counters()
        assert (c["gate_overflows"] == 1) == force
        dispatched[force] = c["backward_tiles_dispatched"] - before
    assert dispatched[True] <= 1.1 * dispatched[False] + 3, dispatched
