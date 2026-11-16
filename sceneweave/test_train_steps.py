"""Batched training steps: equivalence with single steps, the intersection-buffer overflow path, and a static
guard that the step recording path has no fence wait or readback (one fence per batch).

The backward pass uses float atomics, so two identical runs differ by a small run-to-run noise (about 2e-3
in rotations). There is no deterministic mode; every comparison is bounded by 4x the noise measured between
two identical single-step runs (plus a small floor).
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "vksplat" / "src"
STEPS = 12
FLOOR = 1e-4


@pytest.fixture(scope="module")
def dataset():
    with tempfile.TemporaryDirectory() as td:
        yield sc.build_dataset(Path(td) / "ds", n_points=1500)


def _train(ds, out, batch, steps=STEPS, force_capacity=None, force_at=0):
    """Train `steps` steps in batches of `batch` (batch=0: one train_step call per step)."""
    mod = sc.new_trainer(ds, out)
    n_train = int(mod.num_train)
    isects = []
    step = 0
    while step < steps:
        k = max(batch, 1)
        if force_capacity is not None and step == force_at:
            mod._force_isect_capacity(force_capacity)
        slots = [(step + i) % n_train for i in range(min(k, steps - step))]
        if batch == 0:
            mod.train_step(slots[0], step)
        else:
            mod.train_steps(slots, step)
        step += len(slots)
        isects.append(int(mod.get_work_counters()["tile_intersections_last"]))
    return mod, isects


def _loss(mod):
    """Mean L1 of the current render against every training image (the training loss without SSIM)."""
    total = 0.0
    for slot in range(int(mod.num_train)):
        mod.render_train(slot)
        h, w = np.asarray(mod.get_train_image(slot)).shape[:2]
        pred = np.asarray(mod.pixel_state, dtype=np.float32).reshape(h, w, 4)[..., :3]
        ref = np.asarray(mod.get_train_image(slot), dtype=np.float32)[..., :3] / 255.0
        total += float(np.abs(pred - ref).mean())
    return total / int(mod.num_train)


def _stats(mod):
    snap = sc.snapshot(mod)
    return {k: v for k, v in snap.items() if k != "gids"}


def _noise(a: dict, b: dict):
    return {k: float(np.abs(a[k] - b[k]).max()) for k in a}


def _assert_close(a: dict, b: dict, noise: dict, what: str):
    for k in a:
        tol = max(4 * noise[k], FLOOR)
        diff = float(np.abs(a[k] - b[k]).max())
        assert diff <= tol, f"{what}: {k} differs by {diff:.3e} > {tol:.3e}"
        # parameter statistics (mean / std) are far tighter than the max
        assert abs(float(a[k].mean()) - float(b[k].mean())) <= tol
        assert abs(float(a[k].std()) - float(b[k].std())) <= tol


@pytest.fixture(scope="module")
def reference(dataset, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ref")
    mod1, isects = _train(dataset, tmp / "r1", batch=0)
    mod2, _ = _train(dataset, tmp / "r2", batch=0)
    s1, s2 = _stats(mod1), _stats(mod2)
    return {"stats": s1, "noise": _noise(s1, s2), "loss": _loss(mod1), "loss_noise": abs(_loss(mod1) - _loss(mod2)),
            "isects": isects}


@pytest.mark.parametrize("batch", [2, 4])
def test_batched_steps_match_single_steps(dataset, tmp_path, reference, batch):
    mod, _ = _train(dataset, tmp_path / "b", batch=batch)
    _assert_close(_stats(mod), reference["stats"], reference["noise"], f"batch={batch}")
    assert abs(_loss(mod) - reference["loss"]) <= max(4 * reference["loss_noise"], 1e-3)


def test_overflow_path_is_exact(dataset, tmp_path, reference):
    """Buffers too small for one step in the middle of a batch: the gate skips that step and every later step
    of the batch, the host grows the buffers and records them again; the result equals single steps."""
    counts = reference["isects"]
    # batch 2 covers steps 4..7 (step 4 is not a refine step): find a step whose count exceeds all earlier ones
    cap = None
    for p in range(5, 8):
        if counts[p] > max(counts[4:p]) + 50:
            cap = max(counts[4:p]) + 25
            break
    if cap is None:
        pytest.skip("intersection counts of the synthetic scene do not rise inside the batch")
    mod, _ = _train(dataset, tmp_path / "o", batch=4, force_capacity=cap, force_at=4)
    c = mod.get_work_counters()
    assert c["gate_overflows"] >= 1 and c["gate_steps_redone"] >= 1
    assert c["step_buffer_reallocs"] > 0
    assert c["forward_passes"] == STEPS  # every step ran exactly once (skipped steps are not counted)
    _assert_close(_stats(mod), reference["stats"], reference["noise"], "overflow")


def test_overflow_at_first_step_applies_nothing_twice(dataset, tmp_path):
    """Capacity far too small: step 0 of the batch overflows, the whole batch is skipped on the GPU and redone."""
    mod = sc.new_trainer(dataset, tmp_path / "m")
    mod.train_step(0, 0)  # refine step: synchronous path, sizes the buffers
    mod._force_isect_capacity(16)
    mod.train_steps([1, 2, 0], 1)
    c = mod.get_work_counters()
    assert c["gate_overflows"] == 1 and c["gate_steps_redone"] == 3
    assert c["forward_passes"] == 4


def _body(path: Path, name: str, upto: str = None) -> str:
    text = path.read_text()
    m = re.search(rf"^[\w:<>&\*, ]+\b{name}\(", text, re.M)
    assert m, f"{name} not found in {path.name}"
    i, depth = m.end(), 1
    while depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    start = text.index("{", i)
    depth, j = 1, start + 1
    while depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    body = text[start:j]
    return body[: body.index(upto)] if upto else body


FORBIDDEN = ("readElement", "copyFromDevice(", "vkWaitForFences", "vkQueueWaitIdle", "endCommandBatch",
             "HOST_GUARD", "HostGuard", "vkGetFenceStatus", "vkQueueSubmit")


def test_step_recording_path_has_no_fence_or_readback():
    recording = [
        ("python_bindings.cpp", "train_step_gated", None),
        ("gs_renderer.cpp", "executeTilesDeferred", None),
        ("gs_renderer.cpp", "gateBeginBatch", None),
        ("gs_renderer.cpp", "gateBeginStep", None),
        ("gs_renderer.cpp", "gateEndBatch", None),
        ("gs_renderer.cpp", "executeCalculateIndexBufferOffset", "if (!read_count)"),
        ("gs_renderer.cpp", "executeSort", None),
        ("gs_renderer.cpp", "executeComputeTileRanges", None),
        ("gs_renderer.cpp", "executeRasterizeForward", None),
        ("gs_renderer.cpp", "executeRasterizeBackward", None),
        ("gs_trainer.cpp", "executeFusedProjectionBackwardOptimizerStep", None),
    ]
    for fname, func, upto in recording:
        body = _body(SRC / fname, func, upto)
        for token in FORBIDDEN:
            assert token not in body, f"{func} contains {token}"
    # the one allowed wait: the batch's own fence in endCommandBatch, reached through the DeviceGuard of
    # train_steps_gated (the whole batch is recorded inside it)
    gated = _body(SRC / "python_bindings.cpp", "train_steps_gated")
    assert gated.count("DeviceGuard(") == 1


def test_one_fence_per_batch(dataset, tmp_path):
    mod = sc.new_trainer(dataset, tmp_path / "f")
    n_train = int(mod.num_train)
    step = 0
    for _ in range(16):  # warm-up: every backward variant is used once (descriptor updates fence)
        mod.train_steps([step % n_train], step)
        step += 1
    for batch in (1, 2, 4):
        before = mod.get_work_counters()["fence_waits_total"]
        for _ in range(3):
            mod.train_steps([(step + i) % n_train for i in range(batch)], step)
            step += batch
        assert mod.get_work_counters()["fence_waits_total"] - before == 3, f"batch={batch}"
