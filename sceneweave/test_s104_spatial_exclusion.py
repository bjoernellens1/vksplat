"""Phase 11 Track S104 tests: Pre-projection spatial chunk exclusion & active-tile backward raster skip."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VKSPLAT_DIR = REPO_ROOT / "vksplat"
BUILD_DIR = VKSPLAT_DIR / "build"

for p in (str(BUILD_DIR), str(VKSPLAT_DIR), str(REPO_ROOT / "sceneweave")):
    if p not in sys.path:
        sys.path.insert(0, p)

import shader_freshness  # noqa: E402
import vksplat  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Static & CPU-only tests (always run)
# ---------------------------------------------------------------------------


def test_s104_shader_manifest_freshness():
    assert shader_freshness.check(VKSPLAT_DIR / "shader") == []


def test_s104_python_binding_and_funnel_schema():
    assert hasattr(vksplat.VkSplat, "ago_set_group_render_excluded")
    vk = vksplat.VkSplat()
    st = vk.ago_get_stage_telemetry()
    wf = st["workload_funnel"]
    assert "projection_candidates" in wf
    assert "frozen_gaussians" in wf
    assert "frozen_invisible_gaussians" in wf
    assert wf["projection_candidates"] == 0
    assert wf["frozen_invisible_gaussians"] == 0
    assert wf["visible_gaussians"] is None

    expected_counters = {
        "gpu_projected_gaussians",
        "gpu_tile_intersections",
        "gpu_backward_tiles_executed",
        "gpu_backward_tiles_skipped",
    }
    dc = st["device_executed_counters"]
    assert set(dc.keys()) == expected_counters
    assert all(dc[k] is None for k in expected_counters)

    # Enabling profiling before any GPU step must still report None (zero fabrication).
    vk.set_ago_stage_profiling(True)
    st_on = vk.ago_get_stage_telemetry()
    assert all(st_on["device_executed_counters"][k] is None for k in expected_counters)
    assert st_on["workload_funnel"]["visible_gaussians"] is None
    vk.set_ago_stage_profiling(False)

    # S3 envelope fence accounting keys are present even on empty CPU invocation.
    env = vk.ago_execute_multi_batch_gpu_envelope([], first_step=0)
    assert env["cpu_gpu_sync_count"] == 0
    assert env["intermediate_sync_count"] == 0
    assert env["envelope_completion_fence_count"] == 0


def test_s104_shaders_contain_exclusion_and_tile_skip():
    slang_dir = VKSPLAT_DIR / "slang"

    vertex_src = (slang_dir / "vertex_shader.slang").read_text()
    assert "render_exclude_flags" in vertex_src
    assert "GROUP_STATE_EXCLUDED_RENDER" in vertex_src
    assert "is_splat_frozen_invisible" in vertex_src
    assert "out_ago_gpu_counters" in vertex_src
    assert "InterlockedAdd(out_ago_gpu_counters[0], 1u)" in vertex_src
    assert "InterlockedAdd(out_ago_gpu_counters[1], uint32_t(n_tiles))" in vertex_src

    for bwd_name in (
        "alphablend_shader_bwd_per_pixel.slang",
        "alphablend_shader_bwd_per_splat.slang",
        "alphablend_shader_bwd_tensor.slang",
    ):
        src = (slang_dir / bwd_name).read_text()
        assert "groupshared uint tile_has_active_splat;" in src, f"Missing tile_has_active_splat in {bwd_name}"
        assert "if (tile_has_active_splat == 0u)" in src, f"Missing early return in {bwd_name}"
        assert "out_ago_gpu_counters" in src, f"Missing out_ago_gpu_counters in {bwd_name}"
        assert "InterlockedAdd(out_ago_gpu_counters[2], 1u)" in src, f"Missing executed tile counter in {bwd_name}"
        assert "InterlockedAdd(out_ago_gpu_counters[3], 1u)" in src, f"Missing skipped tile counter in {bwd_name}"


# ---------------------------------------------------------------------------
# (b) GPU-marked S104 tests (run serially by coordinator with RUN_GPU_TESTS=1)
# ---------------------------------------------------------------------------


def _median_stage_timings(vk, n_steps: int = 8, start_step: int = 10) -> dict:
    samples = {
        "projection_ms": [],
        "tile_intersect_sort_ms": [],
        "forward_raster_ms": [],
        "backward_raster_ms": [],
        "optimizer_ms": [],
    }
    n_train = int(vk.num_train)
    for i in range(n_steps):
        vk.train_step((start_step + i) % n_train, start_step + i)
        timings = vk.ago_get_stage_telemetry()["stage_timings_ms"]
        for k in samples:
            val = timings[k]
            if val is not None:
                samples[k].append(float(val))
    return {k: float(np.median(v)) for k, v in samples.items() if v}


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU S104 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_s104_invisible_chunk_exclusion_reduces_projection_and_sort(tmp_path):
    """Excluding an 80% invisible chunk drops projection_candidates by 80% and shrinks sort + raster time."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_s104_excl", n_points=4096, w=128, h=96, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_s104_excl", mcmc_protect_structural_groups=True)
    vk.set_external_strategy_mode(True)
    vk.set_ago_stage_profiling(True)

    n = int(len(vk.xyz_ws))
    n_excl = int(round(0.8 * n))
    gids = np.zeros(n, dtype=np.int32)
    gids[n_excl:] = 1
    vk.set_splat_group_ids(gids)
    vk.set_group_state(0, 0)
    vk.set_group_state(1, 0)

    # Warmup
    for s in range(1, 4):
        vk.train_step(s % int(vk.num_train), s)

    wf_full = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_full["gpu_resident_gaussians"] == n
    assert wf_full["projection_candidates"] == n
    assert wf_full["frozen_invisible_gaussians"] == 0
    t_full = _median_stage_timings(vk, n_steps=8, start_step=4)

    # Exclude 80% invisible chunk (group 0)
    vk.ago_set_group_render_excluded(0, True)
    wf_excl = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_excl["frozen_invisible_gaussians"] == n_excl
    assert wf_excl["projection_candidates"] == n - n_excl
    assert wf_excl["frozen_gaussians"] == n_excl

    # Warmup 1 step after state change, then measure
    vk.train_step(0, 12)
    t_excl = _median_stage_timings(vk, n_steps=8, start_step=13)

    full_pipe = t_full["tile_intersect_sort_ms"] + t_full["forward_raster_ms"] + t_full["backward_raster_ms"]
    excl_pipe = t_excl["tile_intersect_sort_ms"] + t_excl["forward_raster_ms"] + t_excl["backward_raster_ms"]
    assert excl_pipe < full_pipe, (
        f"Expected excluding 80% invisible chunk to shrink sort+fwd+bwd time: "
        f"full={t_full}, excl={t_excl}"
    )

    # Restore group 0 and confirm funnel returns to full projection candidates
    vk.ago_set_group_render_excluded(0, False)
    wf_restored = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_restored["frozen_invisible_gaussians"] == 0
    assert wf_restored["projection_candidates"] == n


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU S104 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
@pytest.mark.parametrize("bwd_variant", [0, 1, 2])
def test_gpu_s104_frozen_visible_tile_skip_preserves_exact_mixed_gradients(tmp_path, bwd_variant):
    """Freezing a spatial group skips frozen-only tiles while preserving <1e-4 parity on occluded active splats."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / f"ds_s104_mix_{bwd_variant}", n_points=512, w=96, h=64, structured=True)

    vk_ref = sc.new_trainer(ds, tmp_path / f"out_ref_{bwd_variant}", mcmc_protect_structural_groups=True)
    vk_opt = sc.new_trainer(ds, tmp_path / f"out_opt_{bwd_variant}", mcmc_protect_structural_groups=True)

    for vk in (vk_ref, vk_opt):
        try:
            vk.force_backward_variant(bwd_variant)
        except ValueError as exc:
            pytest.skip(f"backward variant {bwd_variant} unavailable on this device: {exc}")
        vk.set_external_strategy_mode(True)

    xyz0 = np.asarray(vk_ref.xyz_ws, dtype=np.float32).copy()
    assert np.allclose(xyz0, np.asarray(vk_opt.xyz_ws, dtype=np.float32), atol=1e-6)
    # Partition by x coordinate: left half frozen-visible (group 0), right half active (group 1).
    # Boundary tiles are mixed (frozen-visible + active occlusion); far-left tiles are frozen-only.
    med_x = float(np.median(xyz0[:, 0]))
    gids = np.where(xyz0[:, 0] <= med_x, 0, 1).astype(np.int32)
    frozen_mask = gids == 0
    active_mask = gids == 1
    assert np.any(frozen_mask) and np.any(active_mask)

    for vk in (vk_ref, vk_opt):
        vk.set_splat_group_ids(gids)
        vk.set_group_state(0, 2)  # FROZEN_VISIBLE (LOCKED)
        vk.set_group_state(1, 0)  # ACTIVE

    # Reference: disable freeze_skip_backward (full backward pass over all tiles)
    vk_ref.set_freeze_skip_backward(False)
    # Candidate: enable freeze_skip_backward (S104 active-tile early-exit + frozen skip)
    vk_opt.set_freeze_skip_backward(True)

    vk_ref.train_step(0, 1)
    vk_opt.train_step(0, 1)

    ref_xyz = np.asarray(vk_ref.xyz_ws, dtype=np.float32)
    opt_xyz = np.asarray(vk_opt.xyz_ws, dtype=np.float32)

    # 1. Frozen-visible group 0 must not move at all
    assert float(np.max(np.abs(opt_xyz[frozen_mask] - xyz0[frozen_mask]))) == 0.0
    assert float(np.max(np.abs(ref_xyz[frozen_mask] - xyz0[frozen_mask]))) == 0.0

    # 2. Active group 1 (including splats occluded by frozen-visible group 0) must match < 1e-4
    active_diff = float(np.max(np.abs(opt_xyz[active_mask] - ref_xyz[active_mask])))
    assert active_diff < 1e-4, f"Active xyz mismatch for variant {bwd_variant}: max_diff={active_diff}"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU S104 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_s104_device_executed_counters_and_envelope_fences(tmp_path):
    """Phase 12 Lane A: verify real device-side GPU counters, zero fabrication, and S3 fence accounting."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_s104_counters", n_points=2048, w=128, h=96, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_s104_counters", mcmc_protect_structural_groups=True)
    vk.set_external_strategy_mode(True)

    n = int(len(vk.xyz_ws))
    n_excl = int(round(0.8 * n))
    gids = np.zeros(n, dtype=np.int32)
    gids[n_excl:] = 1
    vk.set_splat_group_ids(gids)
    vk.set_group_state(0, 0)
    vk.set_group_state(1, 0)

    # (a) Profiling disabled -> device_executed_counters values must all be None (zero fabrication)
    vk.set_ago_stage_profiling(False)
    vk.train_step(0, 1)
    st_off = vk.ago_get_stage_telemetry()
    assert st_off["instrumentation_enabled"] is False
    assert all(v is None for v in st_off["device_executed_counters"].values())
    assert st_off["workload_funnel"]["visible_gaussians"] is None

    # (b) Profiling enabled -> exact device-side counts + visible_gaussians populated
    vk.set_ago_stage_profiling(True)
    vk.train_step(0, 2)
    st_full = vk.ago_get_stage_telemetry()
    dc_full = st_full["device_executed_counters"]
    assert isinstance(dc_full["gpu_projected_gaussians"], int) and dc_full["gpu_projected_gaussians"] > 0
    assert isinstance(dc_full["gpu_tile_intersections"], int) and dc_full["gpu_tile_intersections"] > 0
    assert isinstance(dc_full["gpu_backward_tiles_executed"], int) and dc_full["gpu_backward_tiles_executed"] > 0
    assert dc_full["gpu_backward_tiles_skipped"] == 0
    assert st_full["workload_funnel"]["visible_gaussians"] == dc_full["gpu_projected_gaussians"]

    # Excluding 80% invisible chunk dramatically drops both gpu_projected_gaussians and gpu_tile_intersections
    vk.ago_set_group_render_excluded(0, True)
    vk.train_step(0, 3)
    st_excl = vk.ago_get_stage_telemetry()
    dc_excl = st_excl["device_executed_counters"]
    assert dc_excl["gpu_projected_gaussians"] > 0
    assert dc_excl["gpu_tile_intersections"] > 0
    assert dc_excl["gpu_projected_gaussians"] < dc_full["gpu_projected_gaussians"] * 0.5, (
        f"Expected dramatic drop in gpu_projected_gaussians: full={dc_full}, excl={dc_excl}"
    )
    assert dc_excl["gpu_tile_intersections"] < dc_full["gpu_tile_intersections"] * 0.5, (
        f"Expected dramatic drop in gpu_tile_intersections: full={dc_full}, excl={dc_excl}"
    )
    assert st_excl["workload_funnel"]["visible_gaussians"] == dc_excl["gpu_projected_gaussians"]

    # Restore group 0 and freeze spatial left half -> gpu_backward_tiles_skipped > 0
    vk.ago_set_group_render_excluded(0, False)
    xyz = np.asarray(vk.xyz_ws, dtype=np.float32)
    med_x = float(np.median(xyz[:, 0]))
    spatial_gids = np.where(xyz[:, 0] <= med_x, 0, 1).astype(np.int32)
    vk.set_splat_group_ids(spatial_gids)
    vk.set_group_state(0, 2)  # FROZEN_VISIBLE (LOCKED)
    vk.set_group_state(1, 0)  # ACTIVE
    vk.set_freeze_skip_backward(True)
    vk.train_step(0, 4)
    dc_frozen = vk.ago_get_stage_telemetry()["device_executed_counters"]
    assert dc_frozen["gpu_backward_tiles_skipped"] > 0, (
        f"Expected frozen spatial half to skip backward tiles: {dc_frozen}"
    )
    assert dc_frozen["gpu_backward_tiles_executed"] > 0

    # (c) S3 multi-batch GPU envelope fence accounting: intermediate_sync_count == 0, completion == 1
    n_train = int(vk.num_train)
    frames = [i % n_train for i in range(4)]
    _ = vk.ago_execute_multi_batch_gpu_envelope(frames, first_step=5, budget_cost_units=180.0, param_mask=31)
    summary = vk.ago_execute_multi_batch_gpu_envelope(frames, first_step=9, budget_cost_units=180.0, param_mask=31)
    assert summary["steps_executed"] == 4
    assert summary["cpu_gpu_sync_count"] == 1
    assert summary["intermediate_sync_count"] == 0
    assert summary["envelope_completion_fence_count"] == 1

    # Disabling profiling resets device_executed_counters back to None
    vk.set_ago_stage_profiling(False)
    st_off2 = vk.ago_get_stage_telemetry()
    assert all(v is None for v in st_off2["device_executed_counters"].values())
    assert st_off2["workload_funnel"]["visible_gaussians"] is None
