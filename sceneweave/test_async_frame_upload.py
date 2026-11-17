"""Asynchronous keyframe upload (append_train_frame / replace_train_frame): equivalence with the blocking path,
ordering against the next train step, a no-fence-wait guard (static + counter) and a 50-upload stress run.

Training uses float atomics, so a comparison between two runs is bounded by 4x the run-to-run noise measured
between two identical runs (plus a floor).
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

BUFFER_CPP = Path(__file__).resolve().parents[1] / "vksplat" / "src" / "buffer.cpp"
H, W = 48, 64


@pytest.fixture(scope="module")
def dataset():
    with tempfile.TemporaryDirectory() as td:
        yield sc.build_dataset(Path(td) / "ds", n_points=1500)


def _pose(k: int) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = (-0.12 * (k % 6), 0.0, -2.0)
    return c2w


def _frame(rng, k: int) -> np.ndarray:
    img = np.empty((H, W, 4), dtype=np.uint8)
    img[..., :3] = sc.structured_image(W, H, 5 * k)
    img[..., 3] = 255
    if k % 2:  # strong contrast between neighbouring frames so a stale image is visible in the result
        img[..., :3] = 255 - img[..., :3]
    return img


def _append(mod, img, k):
    return mod.append_train_frame(img, _pose(k), 0.8 * W, 0.8 * H, W / 2, H / 2)


def _replace(mod, slot, img, k):
    mod.replace_train_frame(slot, img, _pose(k), 0.8 * W, 0.8 * H, W / 2, H / 2)


def _scenario(ds, out, async_upload: bool, replacements: int = 10, seed: int = 3):
    """Append 4 frames, then replace slots round-robin, training a batch of 4 after every upload."""
    mod = sc.new_trainer(ds, out)
    mod.set_async_frame_upload(async_upload)
    mod.clear_train_frames()
    rng = np.random.default_rng(seed)
    sent = {}
    for k in range(4):
        img = _frame(rng, k)
        sent[_append(mod, img, k)] = img
    rng2 = np.random.default_rng(9)
    mod.append_splats_preserving_optimizer(*sc.make_batch(rng2, 800, 11))
    step = 0
    for r in range(replacements):
        slot = r % 4
        img = _frame(rng, 10 + r)
        _replace(mod, slot, img, 10 + r)
        sent[slot] = img
        mod.train_steps([slot, (slot + 1) % 4, slot, (slot + 2) % 4], step)
        step += 4
    return mod, sent


def _compare_bound(a: dict, b: dict, floor=1e-4):
    return {k: float(np.abs(a[k] - b[k]).max()) for k in a if k != "gids"}


def test_async_matches_sync(dataset, tmp_path):
    asy, _ = _scenario(dataset, tmp_path / "a", True)
    asy2, _ = _scenario(dataset, tmp_path / "a2", True)
    syn, _ = _scenario(dataset, tmp_path / "s", False)
    A, A2, S = sc.snapshot(asy), sc.snapshot(asy2), sc.snapshot(syn)
    noise = _compare_bound(A, A2)
    diff = _compare_bound(A, S)
    for k in diff:
        assert diff[k] <= max(1e-4, 4 * noise[k]), (k, diff[k], noise[k])


def test_replaced_slot_is_visible_to_the_next_step_and_readback(dataset, tmp_path):
    mod, sent = _scenario(dataset, tmp_path / "o", True, replacements=7)
    for slot, img in sent.items():
        got = np.asarray(mod.get_train_image(slot))
        assert got.shape == img.shape and np.array_equal(got, img), slot
    # ordering: the replaced image drives the very next step. Train only on slot 0 after replacing it with the
    # inverse image; the async result must match the blocking one (a stale image would move the splats elsewhere).
    results = []
    for async_upload in (True, False, True):
        m = sc.new_trainer(dataset, tmp_path / f"v{len(results)}")
        m.set_async_frame_upload(async_upload)
        rng = np.random.default_rng(4)
        m.append_splats_preserving_optimizer(*sc.make_batch(np.random.default_rng(9), 800, 11))
        m.clear_train_frames()
        _append(m, _frame(rng, 0), 0)
        _replace(m, 0, _frame(rng, 1), 1)
        m.train_steps([0, 0, 0, 0], 0)
        results.append(sc.snapshot(m))
    noise = _compare_bound(results[0], results[2])
    diff = _compare_bound(results[0], results[1])
    for k in diff:
        assert diff[k] <= max(1e-4, 4 * noise[k]), (k, diff[k], noise[k])


def test_no_fence_wait_when_slots_are_free(dataset, tmp_path):
    mod = sc.new_trainer(dataset, tmp_path / "o")
    mod.clear_train_frames()
    rng = np.random.default_rng(1)
    _append(mod, _frame(rng, 0), 0)
    _append(mod, _frame(rng, 1), 1)
    mod.train_steps([0, 1, 0, 1], 0)
    before = mod.get_work_counters()
    for r in range(12):
        _replace(mod, r % 2, _frame(rng, r), r)
        mod.train_steps([r % 2, (r + 1) % 2, r % 2, (r + 1) % 2], 4 + 4 * r)
    after = mod.get_work_counters()
    assert after["frame_upload_waits"] - before["frame_upload_waits"] == 0
    # one fence per train_steps batch, none from the 12 uploads
    assert after["fence_waits_total"] - before["fence_waits_total"] <= 12 + 12, (before, after)


def test_upload_path_has_no_host_wait_statically():
    src = BUFFER_CPP.read_text()
    start = src.index("void VulkanGSPipeline::uploadDeviceAsync(")
    body = re.sub(r"//[^\n]*", "", src[start:src.index("\n}\n", start)])
    forbidden = ["vkWaitForFences", "vkQueueWaitIdle", "vkDeviceWaitIdle", "endCommandBatch", "copyFromDevice",
                 "copyToDevice", "waitUploads", "vkMapMemory", "vkUnmapMemory"]
    hits = [t for t in forbidden if t in body]
    assert not hits, hits
    wait = src[src.index("void VulkanGSPipeline::waitUploadSlot("):]
    assert "vkGetFenceStatus" in wait[:wait.index("\n}\n")]  # a slot is waited on only if still running


def test_stress_50_uploads_interleaved_with_training(dataset, tmp_path):
    mod = sc.new_trainer(dataset, tmp_path / "o")
    mod.clear_train_frames()
    rng = np.random.default_rng(6)
    mod.append_splats_preserving_optimizer(*sc.make_batch(np.random.default_rng(9), 800, 11))
    sent = {}
    n = 0
    step = 0
    for k in range(50):
        img = _frame(rng, k)
        if n < 6:
            sent[_append(mod, img, k)] = img
            n += 1
        else:
            slot = int(rng.integers(0, n))
            _replace(mod, slot, img, k)
            sent[slot] = img
        if k % 3 != 2:  # sometimes two uploads in a row
            mod.train_steps([int(rng.integers(0, n)) for _ in range(4)], step)
            step += 4
    for slot, img in sent.items():
        assert np.array_equal(np.asarray(mod.get_train_image(slot)), img), slot
    # render compare: the render of every slot's camera is finite and the train image readback is intact
    for slot in sent:
        mod.render_train(slot)
        assert np.isfinite(np.asarray(mod.pixel_state, dtype=np.float32)).all()
