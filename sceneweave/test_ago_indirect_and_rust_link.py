"""
Phase 11 Wave 1 Verification Suite (P11-A + P11-B + P11-C + P11-D(GPU)):
  (a) CPU / static checks:
      - Real vkCmdDispatchIndirect in gs_pipeline.cpp + executeComputeIndirect(ago_cutoff_indirect_buf_) in gs_trainer.cpp
      - Dedicated VkQueryPool (VK_QUERY_TYPE_TIMESTAMP) & optional VK_EXT_debug_utils labels in gs_pipeline.cpp
      - Shader freshness (ago_histogram_cutoff.spv + ago_region_vector_priority.spv)
      - Directly-linked Rust libago_control_plane.a native C++ backend parity (ago_scheduler_init(..., backend="rust"))
      - Rust bench_replay 0 hot-path heap allocations check
  (b) GPU-marked tests (@pytest.mark.gpu, run serially by Coordinator with RUN_GPU_TESTS=1):
      - Indirect dispatch + EMPTY -> group_count_x == 0 + dynamic utility-cost cutoff K* monotonicity
      - Multi-batch S3 autonomous GPU envelope: cpu_gpu_sync_count == 1 + VkQueryPool hardware timestamps > 0
      - GPU-resident 6-component RegionPriorityVector & bounded structural candidate selection
      - Live false-color Gaussian rasterizer mode (0..3) with bit-exact SH restore
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "vksplat" / "build"
VKSPLAT_DIR = REPO_ROOT / "vksplat"
SCENEWEAVE_DIR = REPO_ROOT / "sceneweave"
for p in (BUILD_DIR, VKSPLAT_DIR, SCENEWEAVE_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import shader_freshness  # noqa: E402
import vksplat  # noqa: E402


def _strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


# ---------------------------------------------------------------------------
# (a) Static & CPU Runtime Checks
# ---------------------------------------------------------------------------


def test_static_indirect_dispatch_and_gpu_profiling_wired():
    """Verify real vkCmdDispatchIndirect, VkQueryPool timestamps, and VK_EXT_debug_utils labels."""
    pipeline_cpp = _strip_cpp_comments(
        (REPO_ROOT / "vksplat" / "src" / "gs_pipeline.cpp").read_text(encoding="utf-8")
    )
    trainer_cpp = _strip_cpp_comments(
        (REPO_ROOT / "vksplat" / "src" / "gs_trainer.cpp").read_text(encoding="utf-8")
    )

    assert "executeComputeIndirect(" in pipeline_cpp
    assert re.search(
        r"vkCmdDispatchIndirect\s*\(\s*command_buffer\s*,\s*indirect_buf\.buffer\s*,\s*indirect_offset\s*\)",
        pipeline_cpp,
    ), "executeComputeIndirect must call vkCmdDispatchIndirect(command_buffer, indirect_buf.buffer, indirect_offset)"

    assert "VK_QUERY_TYPE_TIMESTAMP" in pipeline_cpp
    assert "ago_timestamp_query_pool_" in pipeline_cpp
    assert "VK_EXT_DEBUG_UTILS_EXTENSION_NAME" in pipeline_cpp
    assert "vkCmdBeginDebugUtilsLabelEXT" in pipeline_cpp

    assert re.search(
        r"executeComputeIndirect\s*\(\s*ago_cutoff_indirect_buf_\.deviceBuffer",
        trainer_cpp,
    ), "gs_trainer.cpp must dispatch via executeComputeIndirect(ago_cutoff_indirect_buf_.deviceBuffer, ...)"


def test_shader_manifest_freshness():
    """All compiled SPIR-V shaders (including ago_histogram_cutoff & ago_region_vector_priority) must be fresh."""
    stale = shader_freshness.check(REPO_ROOT / "vksplat" / "shader", strict=False)
    assert stale == [], f"Stale shaders detected: {stale}"
    assert (REPO_ROOT / "vksplat" / "shader" / "generated" / "ago_histogram_cutoff.spv").is_file()
    assert (REPO_ROOT / "vksplat" / "shader" / "generated" / "ago_region_vector_priority.spv").is_file()


def test_rust_staticlib_native_cpp_backend_parity():
    """Verify VkSplat links libago_control_plane.a directly and runs backend='rust' with zero Python ctypes."""
    assert vksplat.VkSplat.has_rust_control_plane() is True

    vk_cpp = vksplat.VkSplat()
    vk_rust = vksplat.VkSplat()

    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 1
    vk_cpp.ago_scheduler_init(cfg, use_gpu_executor=False, backend="cpp")
    vk_rust.ago_scheduler_init(cfg, use_gpu_executor=False, backend="rust")

    assert vk_cpp.ago_scheduler_backend() == "cpp"
    assert vk_rust.ago_scheduler_backend() == "rust"

    budget = vksplat.AgoDeadlineBudget()
    budget.now_ms = 0.0
    budget.next_deadline_ms = 12.5
    budget.reserved_ms = 0.0
    budget.inflight_ms = 0.0
    budget.safety_margin_ms = 0.5
    vk_cpp.ago_scheduler_set_budget(budget)
    vk_rust.ago_scheduler_set_budget(budget)

    # Push pose revision bumping pose_version 1 -> 2
    ev = vksplat.AgoEvidenceMessage()
    ev.kind = vksplat.AgoEvidenceKind.POSE_REVISION
    ev.critical = True
    ev.region_id = 7
    ev.pose_version = 2
    ev.timestamp_ms = 100.0
    ev.magnitude = 2.5
    assert vk_cpp.ago_scheduler_push_evidence(ev) is True
    assert vk_rust.ago_scheduler_push_evidence(ev) is True

    # Enqueue: stale P2 (pose_version=1), fresh P2 (pose_version=2), and P0 consistency action
    def _make_action(aid, cls, op, pose_v, gain, cost):
        a = vksplat.AgoAction()
        a.action_id = aid
        a.region_id = 7
        a.service_class = cls
        a.op = op
        a.requires_gpu = True
        a.required_pose_version = pose_v
        a.expected_gain = gain
        a.estimated_cost_ms = cost
        return a

    actions = [
        _make_action(1, vksplat.AgoServiceClass.P2_REFINEMENT, vksplat.AgoOperationType.REFINE, 1, 999.0, 0.5),
        _make_action(2, vksplat.AgoServiceClass.P2_REFINEMENT, vksplat.AgoOperationType.REFINE, 2, 100.0, 0.5),
        _make_action(3, vksplat.AgoServiceClass.P0_CONSISTENCY, vksplat.AgoOperationType.RETRACT, 2, 0.1, 0.5),
    ]
    for a in actions:
        assert vk_cpp.ago_scheduler_enqueue_action(a) is True
        assert vk_rust.ago_scheduler_enqueue_action(a) is True

    tick_cpp = vk_cpp.ago_scheduler_tick()
    tick_rust = vk_rust.ago_scheduler_tick()

    assert tick_cpp.stale_actions_pruned == 1
    assert tick_rust.stale_actions_pruned == 1
    assert tick_cpp.actions_admitted == 1
    assert tick_rust.actions_admitted == 1

    sum_cpp = vk_cpp.ago_scheduler_summary()
    sum_rust = vk_rust.ago_scheduler_summary()
    assert sum_cpp.pose_version == sum_rust.pose_version == 2
    assert sum_cpp.total_rejected_stale == sum_rust.total_rejected_stale == 1
    assert sum_cpp.total_admitted == sum_rust.total_admitted == 1


def test_rust_bench_replay_zero_hot_path_allocations():
    """Run Rust bench_replay binary and verify 0 hot-path heap allocations."""
    bench_bin = REPO_ROOT / "vksplat" / "ago_control_plane" / "target" / "release" / "bench_replay"
    assert bench_bin.is_file(), f"bench_replay binary missing: {bench_bin}"
    proc = subprocess.run([str(bench_bin)], capture_output=True, text=True, check=True)
    assert "hot_allocs=0" in proc.stdout, f"Expected hot_allocs=0 in output: {proc.stdout}"
    assert "hot_bytes=0" in proc.stdout, f"Expected hot_bytes=0 in output: {proc.stdout}"


def test_s93_and_s100_cpu_bindings_and_default_telemetry():
    """Verify S93 memory telemetry & S100 stage profiling bindings exist and never fabricate timings."""
    vk = vksplat.VkSplat()
    assert hasattr(vk, "ago_offload_snapshot_to_ram")
    assert hasattr(vk, "ago_restore_snapshot_from_ram")
    assert hasattr(vk, "ago_get_memory_telemetry")
    assert hasattr(vk, "set_ago_stage_profiling")
    assert hasattr(vk, "get_ago_stage_profiling")
    assert hasattr(vk, "ago_get_stage_telemetry")

    mem = vk.ago_get_memory_telemetry()
    assert set(mem.keys()) >= {
        "gpu_device_allocated_bytes",
        "host_snapshot_bytes",
        "staging_bytes",
        "is_gpu_evicted",
        "rss_bytes",
    }
    assert mem["gpu_device_allocated_bytes"] == 0
    assert mem["host_snapshot_bytes"] == 0
    assert mem["staging_bytes"] == 0
    assert mem["is_gpu_evicted"] is False
    assert mem["rss_bytes"] > 0

    assert vk.get_ago_stage_profiling() is False
    st = vk.ago_get_stage_telemetry()
    assert st["instrumentation_enabled"] is False
    expected_stages = {
        "projection_ms",
        "tile_intersect_sort_ms",
        "forward_raster_ms",
        "backward_raster_ms",
        "optimizer_ms",
        "ago_selection_ms",
    }
    assert set(st["stage_timings_ms"].keys()) == expected_stages
    assert all(st["stage_timings_ms"][k] is None for k in expected_stages)

    expected_funnel = {
        "total_stored_gaussians",
        "gpu_resident_gaussians",
        "projection_candidates",
        "visible_gaussians",
        "backward_active_gaussians",
        "geometry_updated_gaussians",
        "frozen_gaussians",
    }
    assert set(st["workload_funnel"].keys()) == expected_funnel
    assert st["workload_funnel"]["visible_gaussians"] is None

    vk.set_ago_stage_profiling(True)
    assert vk.get_ago_stage_profiling() is True
    st_enabled = vk.ago_get_stage_telemetry()
    assert st_enabled["instrumentation_enabled"] is True
    assert all(st_enabled["stage_timings_ms"][k] is None for k in expected_stages)
    vk.set_ago_stage_profiling(False)
    assert vk.get_ago_stage_profiling() is False


# ---------------------------------------------------------------------------
# (b) GPU-Marked Phase 11 Tests (Coordinator runs serially with RUN_GPU_TESTS=1)
# ---------------------------------------------------------------------------


def _make_warmed_trainer(tmp_path, name: str, n_points: int = 512):
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / f"ds_{name}", n_points=n_points, w=64, h=48, structured=True)
    vk = sc.new_trainer(ds, tmp_path / f"out_{name}", mcmc_protect_structural_groups=True)
    vk.force_backward_variant(1)
    vk.set_external_strategy_mode(True)
    vk.set_ago_telemetry_enabled(True)

    n = int(len(vk.xyz_ws))
    half = n // 2
    gids = np.zeros(n, dtype=np.int32)
    gids[half:] = 1
    vk.set_splat_group_ids(gids)
    vk.set_group_state(0, 0)
    vk.set_group_state(1, 0)

    # Warmup 2 steps so telemetry, Adam moments, and pipeline descriptor sets are initialized
    for s in range(1, 3):
        vk.train_step(s % int(vk.num_train), s)
    return vk


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU Phase 11 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_indirect_dispatch_empty_and_dynamic_k_star(tmp_path):
    """Verify EMPTY -> group_count_x == 0 and dynamic utility-cost cutoff K* monotonicity on GPU."""
    vk = _make_warmed_trainer(tmp_path, "p11_indirect")

    # 1. Exact-K empty set -> group_count_x == 0
    vk.select_ago_active_set_gpu_async(target_fraction=0.0, selection_mode=0, budget_cost_units=0.0)
    st_empty = vk.get_ago_cutoff_indirect_state()
    assert st_empty["group_count_x"] == 0
    assert st_empty["selected_count"] == 0
    assert st_empty["active_set_state"] == 0  # EMPTY

    # 2. Dynamic utility-cost cutoff (selection_mode=1): budget=0 -> EMPTY
    vk.select_ago_active_set_gpu_async(target_fraction=1.0, selection_mode=1, budget_cost_units=0.0)
    st_zero_b = vk.get_ago_cutoff_indirect_state()
    assert st_zero_b["group_count_x"] == 0
    assert st_zero_b["selected_count"] == 0

    # 3. Small vs larger cost budget -> monotonic K* and valid indirect dispatch group_count_x > 0
    vk.select_ago_active_set_gpu_async(
        target_fraction=1.0, newborn_warmup_steps=0.0, selection_mode=1, budget_cost_units=64.0
    )
    st_lo = vk.get_ago_cutoff_indirect_state()
    vk.select_ago_active_set_gpu_async(
        target_fraction=1.0, newborn_warmup_steps=0.0, selection_mode=1, budget_cost_units=256.0
    )
    st_hi = vk.get_ago_cutoff_indirect_state()

    assert st_lo["selected_count"] > 0
    assert st_lo["group_count_x"] > 0
    assert st_hi["selected_count"] >= st_lo["selected_count"]
    assert st_hi["group_count_x"] >= st_lo["group_count_x"]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU Phase 11 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_s3_multi_batch_envelope_and_timestamps(tmp_path):
    """Verify multi-batch S3 executes in 1 CPU-GPU sync with VkQueryPool hardware GPU timestamps > 0."""
    vk = _make_warmed_trainer(tmp_path, "p11_s3_batch")
    n_train = int(vk.num_train)
    frames = [i % n_train for i in range(4)]

    # Run once to ensure gated pipelines have bound current descriptors, then verify steady-state 1 sync.
    _ = vk.ago_execute_multi_batch_gpu_envelope(frames, first_step=3, budget_cost_units=180.0, param_mask=31)
    summary = vk.ago_execute_multi_batch_gpu_envelope(frames, first_step=7, budget_cost_units=180.0, param_mask=31)

    assert summary["steps_executed"] == 4
    assert summary["cpu_gpu_sync_count"] == 1
    assert summary["indirect_dispatch_used"] is True
    assert summary["overflow"] is False
    if summary["timestamp_queries_enabled"]:
        assert summary["gpu_selection_ms"] > 0.0
        assert summary["gpu_active_ms"] > 0.0
        assert summary["gpu_active_ms"] >= summary["gpu_selection_ms"]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU Phase 11 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_region_priority_vectors_and_structural_candidates(tmp_path):
    """Verify GPU-resident 6-component RegionPriorityVector and bounded split/prune candidates."""
    vk = _make_warmed_trainer(tmp_path, "p11_region_prio")
    n = int(len(vk.xyz_ws))

    vecs = vk.get_ago_region_priority_vectors()
    assert len(vecs) == 2
    region_ids = sorted(v["region_id"] for v in vecs)
    assert region_ids == [0, 1]
    for v in vecs:
        for key in ("u_geom", "u_app", "u_split", "u_birth", "u_merge", "u_corr"):
            assert np.isfinite(v[key]) and v[key] >= 0.0
        assert 0 <= v["dominant_op"] <= 5
        assert v["splat_count"] > 0
        assert (v["eligibility_flags"] & 1) != 0

    cands = vk.get_ago_structural_candidates(max_split=16, max_prune=16, split_scale_thresh=0.0, split_grad_thresh=0.0, prune_opacity_thresh=1.0)
    assert len(cands["split_candidate_indices"]) <= 16
    assert len(cands["prune_candidate_indices"]) <= 16
    assert cands["split_candidates_total"] > 0
    assert cands["prune_candidates_total"] > 0
    assert all(0 <= idx < n for idx in cands["split_candidate_indices"])
    assert all(0 <= idx < n for idx in cands["prune_candidate_indices"])


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU Phase 11 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_false_color_render_and_bit_exact_restore(tmp_path):
    """Verify live false-color Gaussian rasterizer (modes 0..3) and bit-exact SH/param restoration."""
    vk = _make_warmed_trainer(tmp_path, "p11_false_color")

    sh_before = np.array(vk.sh_coeffs, copy=True)
    xyz_before = np.array(vk.xyz_ws, copy=True)
    rot_before = np.array(vk.rotations, copy=True)
    scales_before = np.array(vk.scales, copy=True)
    opac_before = np.array(vk.opacities, copy=True)

    expected_names = ["RESIDUAL", "PRIORITY", "GPU_WORK", "OUTCOME"]
    for mode, expected_name in enumerate(expected_names):
        res = vk.render_ago_false_color(0, mode=mode)
        assert res["mode"] == mode
        assert res["mode_name"] == expected_name
        rgb = np.array(res["rgb"], copy=False)
        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        assert np.all(np.isfinite(rgb))

    assert np.array_equal(np.array(vk.sh_coeffs, copy=True), sh_before), "sh_coeffs not restored bit-for-bit!"
    assert np.array_equal(np.array(vk.xyz_ws, copy=True), xyz_before), "xyz_ws mutated!"
    assert np.array_equal(np.array(vk.rotations, copy=True), rot_before), "rotations mutated!"
    assert np.array_equal(np.array(vk.scales, copy=True), scales_before), "scales mutated!"
    assert np.array_equal(np.array(vk.opacities, copy=True), opac_before), "opacities mutated!"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU Phase 11 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_s93_gpu_eviction_and_immediate_post_restore_render(tmp_path):
    """S93: Verify real GPU buffer eviction drops gpu_device_allocated_bytes and immediate post-restore render matches."""
    vk = _make_warmed_trainer(tmp_path, "p11_s93_evict")
    n_splats = int(len(vk.xyz_ws))
    assert n_splats > 0

    ref_render = vk.render_val_depth(0)
    ref_rgb = np.array(ref_render["rgb"], copy=True)
    ref_depth = np.array(ref_render["depth_expected"], copy=True)

    mem_before = vk.ago_get_memory_telemetry()
    assert mem_before["is_gpu_evicted"] is False
    assert mem_before["gpu_device_allocated_bytes"] > 0
    assert mem_before["rss_bytes"] > 0

    snap_id = vk.ago_offload_snapshot_to_ram(evict_gpu=True)
    assert snap_id > 0

    mem_evicted = vk.ago_get_memory_telemetry()
    assert mem_evicted["is_gpu_evicted"] is True
    assert mem_evicted["host_snapshot_bytes"] > 0
    assert mem_evicted["gpu_device_allocated_bytes"] < mem_before["gpu_device_allocated_bytes"] * 0.25

    wf_evicted = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_evicted["gpu_resident_gaussians"] == 0
    assert wf_evicted["total_stored_gaussians"] == n_splats

    assert vk.ago_restore_snapshot_from_ram(snap_id) is True

    mem_restored = vk.ago_get_memory_telemetry()
    assert mem_restored["is_gpu_evicted"] is False
    assert mem_restored["gpu_device_allocated_bytes"] >= mem_before["gpu_device_allocated_bytes"] * 0.9

    # Immediate render_val_depth(0) BEFORE any extra optimization steps must match ref_render
    post_render = vk.render_val_depth(0)
    post_rgb = np.array(post_render["rgb"], copy=True)
    post_depth = np.array(post_render["depth_expected"], copy=True)
    assert float(np.max(np.abs(post_rgb - ref_rgb))) < 1e-4
    assert float(np.max(np.abs(post_depth - ref_depth))) < 1e-4

    # Subsequent training step must also work cleanly after re-allocation
    vk.train_step(0, 3)
    assert vk.ago_drop_ram_snapshot(snap_id) is True
    assert vk.ago_get_memory_telemetry()["host_snapshot_bytes"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU Phase 11 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_s100_stage_telemetry_and_workload_funnel(tmp_path):
    """S100: Verify opt-in per-stage Vulkan timestamps (None when disabled, >0 when enabled) and workload funnel."""
    vk = _make_warmed_trainer(tmp_path, "p11_s100_stages")
    n_splats = int(len(vk.xyz_ws))

    # 1. Profiling disabled -> all stage timings must be None (never fabricated 0.0)
    vk.set_ago_stage_profiling(False)
    vk.train_step(0, 3)
    st_off = vk.ago_get_stage_telemetry()
    assert st_off["instrumentation_enabled"] is False
    assert all(v is None for v in st_off["stage_timings_ms"].values())

    # 2. Profiling enabled -> real >0 Vulkan stage timestamps on training step + AGO selection
    vk.set_ago_stage_profiling(True)
    vk.select_ago_active_set_gpu_async(target_fraction=1.0, newborn_warmup_steps=0.0, selection_mode=0)
    vk.train_step(0, 4)
    st_on = vk.ago_get_stage_telemetry()
    assert st_on["instrumentation_enabled"] is True
    for k in (
        "projection_ms",
        "tile_intersect_sort_ms",
        "forward_raster_ms",
        "backward_raster_ms",
        "optimizer_ms",
        "ago_selection_ms",
    ):
        val = st_on["stage_timings_ms"][k]
        assert val is not None and np.isfinite(val) and val > 0.0, f"Expected real >0 ms for {k}, got {val}"

    # 3. Workload funnel: PARAM_SH (16) -> geometry_updated_gaussians == 0
    vk.set_ago_param_mask(16)
    wf_sh = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_sh["total_stored_gaussians"] == n_splats
    assert wf_sh["gpu_resident_gaussians"] == n_splats
    assert wf_sh["projection_candidates"] == n_splats
    assert wf_sh["frozen_gaussians"] == 0
    assert wf_sh["backward_active_gaussians"] == n_splats
    assert wf_sh["geometry_updated_gaussians"] == 0

    # Restore PARAM_ALL (31) -> geometry_updated_gaussians == backward_active_gaussians
    vk.set_ago_param_mask(31)
    wf_all = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_all["geometry_updated_gaussians"] == wf_all["backward_active_gaussians"] == n_splats

    # 4. Freeze group 0 (half of splats) -> frozen_gaussians increases, backward_active_gaussians shrinks
    assert vk.ago_execute_external_op(vksplat.AgoOperationType.FREEZE, 0) == vksplat.AgoCompletionStatus.OK
    wf_frozen = vk.ago_get_stage_telemetry()["workload_funnel"]
    assert wf_frozen["frozen_gaussians"] == n_splats // 2
    assert wf_frozen["backward_active_gaussians"] == n_splats - wf_frozen["frozen_gaussians"]
    assert wf_frozen["geometry_updated_gaussians"] == wf_frozen["backward_active_gaussians"]

    # 5. Disabling profiling resets timings back to None
    vk.set_ago_stage_profiling(False)
    st_off2 = vk.ago_get_stage_telemetry()
    assert st_off2["instrumentation_enabled"] is False
    assert all(v is None for v in st_off2["stage_timings_ms"].values())
