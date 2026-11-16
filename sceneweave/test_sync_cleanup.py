"""Per-frame sync cleanup: GPU scale clamp, masked retraction, fence-wait budget, no-readback guard."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_scenario as sc  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "vksplat" / "src"
MAX_S, TANGENT_CAP = 0.03, 0.045


def host_clamp(s_norm, gids, max_s_norm, tangent_cap):
    """Port of StreamingMappingSession._clamp_runaway_scales_on_module (limits part, operating in module units)."""
    limits = np.full_like(s_norm, max_s_norm, dtype=np.float32)
    if len(gids) == len(s_norm):
        limits[gids >= 0, 0:2] = tangent_cap
    return np.minimum(s_norm, limits).astype(np.float32)


def _build(tmp_path, n_batches=10):
    ds = sc.build_dataset(tmp_path / "ds", n_points=300)
    mod = sc.new_trainer(ds, tmp_path / "out")
    rng = np.random.default_rng(5)
    for i in range(n_batches):
        xyz, sh, rot, scales, op, gids = sc.make_batch(rng, 40, 10 + i if i % 2 else -1)
        scales = (scales * rng.uniform(0.2, 3.0, size=scales.shape)).astype(np.float32)
        mod.append_splats_preserving_optimizer(xyz, sh, rot, scales, op, gids)
    return mod


def test_clamp_scales_matches_host_logic(tmp_path):
    mod = _build(tmp_path)
    before = sc.snapshot(mod)
    assert (before["scales"] > MAX_S).any() and (before["scales"][before["gids"] >= 0] > TANGENT_CAP).any()
    expected = host_clamp(before["scales"], before["gids"], MAX_S, TANGENT_CAP)
    mod.clamp_scales(MAX_S, TANGENT_CAP)
    after = sc.snapshot(mod)
    assert np.abs(after["scales"] - expected).max() <= 1e-7
    for k in ("xyz", "rot", "opac", "sh", "gids"):
        assert np.array_equal(before[k], after[k]), k
    # idempotent
    mod.clamp_scales(MAX_S, TANGENT_CAP)
    assert np.array_equal(sc.snapshot(mod)["scales"], after["scales"])


def test_masked_retraction_leaves_other_splats_untouched(tmp_path):
    mod = _build(tmp_path)
    before = sc.snapshot(mod)
    gid = 11
    m = before["gids"] == gid
    assert m.any()
    mod.retract_group_sparse(gid)
    after = sc.snapshot(mod)
    assert np.all(after["opac"][m] == 0.0)
    for k in ("xyz", "rot", "scales", "sh", "gids"):
        assert np.array_equal(before[k], after[k]), k
    assert np.array_equal(before["opac"][~m], after["opac"][~m])


def _body(text, signature):
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    for j in range(i, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            return text[start:j + 1]
    raise AssertionError(signature)


def test_clamp_and_retraction_paths_have_no_readback():
    trainer = (SRC / "gs_trainer.cpp").read_text()
    bindings = (SRC / "python_bindings.cpp").read_text()
    forbidden = re.compile(r"copyFromDevice|downloadDeviceSlice|buffer_to_array|vkWaitForFences|HOST_GUARD|vkQueueWaitIdle")
    for text, sig in (
        (trainer, "void VulkanGSTrainer::clamp_scales("),
        (trainer, "void VulkanGSTrainer::retract_group_sparse("),
        (bindings, "void clamp_scales("),
        (bindings, "void retract_group_sparse("),
    ):
        assert not forbidden.search(_body(text, sig)), sig


def test_fence_waits_per_frame(tmp_path):
    """10 appends + clamps + 1 retraction. The GPU paths must need far fewer blocking fence waits than readback paths."""
    mod = _build(tmp_path, n_batches=0)
    rng = np.random.default_rng(9)
    fw = lambda: int(mod.get_work_counters()["fence_waits_total"])  # noqa: E731

    def scenario(clamp, retract):
        w0 = fw()
        for i in range(10):
            mod.append_splats_preserving_optimizer(*sc.make_batch(rng, 32, 10 + i % 3))
            if i % 2 == 1:
                clamp()
        retract()
        return fw() - w0

    def host_clamp_call():  # what the session did: three readbacks and one upload
        packed = np.asarray(mod.get_scales_opacities_packed(), dtype=np.float32).reshape(-1, 4).copy()
        gids = np.asarray(mod.get_splat_group_ids(), dtype=np.int32).reshape(-1)
        packed[:, :3] = host_clamp(packed[:, :3], gids, MAX_S, TANGENT_CAP)
        mod._set_scales_opacs(np.ascontiguousarray(packed))

    def host_retract():  # full readback of the session mirror, then group update
        for name in ("xyz_ws", "scales", "rotations", "opacities", "sh_coeffs"):
            np.asarray(getattr(mod, name))
        mod.get_splat_group_ids()

    old = scenario(host_clamp_call, host_retract)
    new = scenario(lambda: mod.clamp_scales(MAX_S, TANGENT_CAP), lambda: mod.retract_group_sparse(11))
    print(f"fence_waits per scenario (10 appends, 5 clamps, 1 retraction): before={old} after={new}")
    assert new < old


def _train_with_variant(tmp_path, variant, steps=10):
    ds = sc.build_dataset(tmp_path / f"ds{variant}", n_points=3000)
    mod = sc.new_trainer(ds, tmp_path / f"out{variant}")
    mod.force_backward_variant(variant)
    n = int(mod.num_train)
    for s in range(1, steps + 1):
        mod.train_step(s % n, s)
    return sc.snapshot(mod)


def test_backward_variants_agree_within_atomic_noise(tmp_path):
    """PerSplat (1) vs Tensor_0_8_8 (3): measured max |diff| after 10 steps is xyz 2.6e-4, scales 1.4e-4,
    opacity 1.2e-3, sh 5e-4, the same order as Tensor's own run-to-run atomic noise (xyz 3e-5, scales 1.5e-4,
    opacity 4e-4, sh 3e-4). That justifies pinning the variant after the exploration phase."""
    sys.path.insert(0, str(sc.ROOT))
    from simple_trainer import TrainerConfig

    assert TrainerConfig().rasterize_backward_pin_after == 20
    a = _train_with_variant(tmp_path, 1)
    b = _train_with_variant(tmp_path, 3)
    assert len(a["gids"]) == len(b["gids"])
    bounds = {"xyz": 1e-3, "scales": 1e-3, "opac": 5e-3, "sh": 3e-3}
    for k, bound in bounds.items():
        assert np.abs(a[k] - b[k]).max() < bound, k
