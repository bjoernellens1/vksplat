"""Reserved-capacity append: equivalence, geometric growth, group sync and a static no-fence guard."""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

GS_TRAINER = Path(__file__).resolve().parents[1] / "vksplat" / "src" / "gs_trainer.cpp"


def _assert_same(a: dict, b: dict, tol: float = 0.0):
    for key in a:
        assert a[key].shape == b[key].shape, key
        if tol == 0.0:
            assert np.array_equal(a[key], b[key]), f"{key} differs (max {np.abs(a[key] - b[key]).max()})"
        else:
            assert np.abs(a[key] - b[key]).max() <= tol, key


def test_k_batches_equal_single_big_append(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=1000)
    kw = dict(batches=5, m=21, steps_between=0)
    # no training: appended parameters (incl. the reordered SH blocks and group ids) are bit-identical
    _assert_same(sc.run(ds, tmp_path / "o1", final_steps=0, **kw),
                 sc.run(ds, tmp_path / "o2", final_steps=0, merge=True, **kw))
    # with training the GPU backward is not bitwise reproducible (atomics): bound by the run-to-run noise
    many = sc.run(ds, tmp_path / "o3", final_steps=6, **kw)
    one = sc.run(ds, tmp_path / "o4", final_steps=6, merge=True, **kw)
    again = sc.run(ds, tmp_path / "o5", final_steps=6, **kw)
    noise = max(np.abs(many[k] - again[k]).max() for k in many)
    assert np.array_equal(many["gids"], one["gids"])
    _assert_same(many, one, max(1e-6, 4 * noise))


@pytest.mark.skipif("VKSPLAT_BASELINE_DIR" not in os.environ, reason="needs a vksplat build at fd1a50d")
def test_matches_baseline_build(tmp_path):
    """Same scenario on the baseline build (VKSPLAT_BASELINE_DIR: python files + old .so) and on this build.
    Training is not bitwise reproducible, so the tolerance is 4x the baseline's own run-to-run noise."""
    outs = {}
    for name, base in (("new", ""), ("old", os.environ["VKSPLAT_BASELINE_DIR"]), ("old2", os.environ["VKSPLAT_BASELINE_DIR"])):
        dest = tmp_path / f"{name}.npz"
        subprocess.run([sys.executable, str(Path(sc.__file__)), str(dest)], check=True,
                       env=dict(os.environ, VKSPLAT_PY_DIR=base))
        outs[name] = dict(np.load(dest))
    noise = max(np.abs(outs["old"][k] - outs["old2"][k]).max() for k in outs["old"])
    assert np.array_equal(outs["new"]["gids"], outs["old"]["gids"])
    _assert_same(outs["new"], outs["old"], max(1e-6, 4 * noise))


def test_capacity_grows_geometrically(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=5000)
    mod = sc.new_trainer(ds, tmp_path / "out")
    n_train = int(mod.num_train)
    rng = np.random.default_rng(1)
    n0 = len(np.asarray(mod.xyz_ws))
    before = mod.get_work_counters()
    caps = [mod.get_splat_capacity()["xyz_ws"]]
    for i in range(50):
        mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 100, i))
        mod.train_step(i % n_train, i)
        caps.append(mod.get_splat_capacity()["xyz_ws"])
    after = mod.get_work_counters()
    assert len(np.asarray(mod.xyz_ws)) == n0 + 5000
    assert all(c >= n0 + 100 * i for i, c in enumerate(caps))
    grown = sorted(set(caps))
    assert all(b >= 1.25 * a for a, b in zip(grown, grown[1:])), grown
    reallocs = after["append_reallocs"] - before["append_reallocs"]
    assert 1 <= reallocs <= 8, reallocs
    assert reallocs >= len(grown) - 1
    assert after["append_fence_waits"] - before["append_fence_waits"] == 0
    assert after["step_buffer_reallocs"] - before["step_buffer_reallocs"] <= 3 * 8 * 11, after


def test_group_ids_and_state_stay_in_sync_after_growth(tmp_path):
    ds = sc.build_dataset(tmp_path / "ds", n_points=300)
    mod = sc.new_trainer(ds, tmp_path / "out")
    n_train = int(mod.num_train)
    rng = np.random.default_rng(2)
    before = mod.get_work_counters()["append_reallocs"]
    for g in range(12):
        mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 64, 3 * g))
        mod.train_step(g % n_train, g)
        if g == 4:
            mod.set_group_state(6, 2)
    assert mod.get_work_counters()["append_reallocs"] - before >= 2
    host = np.asarray(mod.get_splat_group_ids())
    assert np.array_equal(np.asarray(mod.get_device_group_ids()), host)
    state = np.asarray(mod.get_device_group_state())
    assert len(state) >= int(host.max()) + 1
    expect = np.ones(len(state), dtype=np.uint32)
    expect[6] = 2
    assert np.array_equal(state, expect)


def test_append_body_has_no_fence_wait_or_readback():
    src = GS_TRAINER.read_text()
    start = src.index("size_t VulkanGSTrainer::append_splats_preserving_optimizer(")
    end = src.index("\n}\n", start)
    body = re.sub(r"//[^\n]*", "", src[start:end])
    forbidden = [
        "vkWaitForFences", "vkQueueWaitIdle", "vkDeviceWaitIdle", "endCommandBatch", "copyToHost",
        "copyFromDevice", "copyToDevice", "downloadDeviceSlice", "uploadDeviceSlice", "readElement",
        "resizeAndCopyDeviceBuffer", "DEVICE_GUARD", "undoReorderSH",
    ]
    hits = [tok for tok in forbidden if tok in body]
    assert not hits, f"append body must stay fence- and readback-free: {hits}"
