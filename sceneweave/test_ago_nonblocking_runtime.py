"""
CPU verification, static non-blocking invariant audit, 11 fault-injection tests,
sub-100us tick() latency benchmark, and GPU parity test (marked @pytest.mark.gpu)
for the VkSplat Native Non-Blocking AGO Control Plane (P1 + P2 + P3 + AGO-0).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "vksplat" / "build"
if BUILD_DIR.exists() and str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import vksplat  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Static Regex Check Banning Blocking Calls
# ---------------------------------------------------------------------------

BANNED_CONTROL_PLANE_PATTERNS = [
    r"\bvkWaitForFences\b",
    r"\bvkQueueWaitIdle\b",
    r"\bvkDeviceWaitIdle\b",
    r"\bvkWaitSemaphores\b",
    r"\bfuture\.get\s*\(",
    r"\bthread\.join\s*\(",
    r"\bHOST_GUARD\b",
]


def _strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _extract_cpp_function_body(text: str, signature_substr: str) -> str:
    idx = text.find(signature_substr)
    assert idx != -1, f"Function signature not found: {signature_substr}"
    brace_start = text.find("{", idx)
    assert brace_start != -1, f"Opening brace not found for {signature_substr}"
    depth = 0
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]
    raise AssertionError(f"Unbalanced braces for {signature_substr}")


def test_static_no_blocking_calls_in_ago_control_plane_and_runtime():
    """MANDATORY INVARIANT: ZERO blocking waits or HOST_GUARD in ago/core, ago/spatial, ago/runtime."""
    target_dirs = [
        REPO_ROOT / "vksplat" / "src" / "ago" / "core",
        REPO_ROOT / "vksplat" / "src" / "ago" / "spatial",
        REPO_ROOT / "vksplat" / "src" / "ago" / "runtime",
    ]
    checked_files: List[Path] = []
    violations: List[str] = []

    for d in target_dirs:
        assert d.exists(), f"Expected directory missing: {d}"
        for path in sorted(d.rglob("*")):
            if path.suffix not in {".h", ".hpp", ".cpp", ".cc"}:
                continue
            checked_files.append(path)
            code = _strip_cpp_comments(path.read_text(encoding="utf-8"))
            for pat in BANNED_CONTROL_PLANE_PATTERNS:
                if re.search(pat, code):
                    violations.append(f"{path.relative_to(REPO_ROOT)}: matched {pat}")

    assert len(checked_files) >= 6, f"Expected >= 6 AGO C++ source/header files, found {len(checked_files)}"
    assert not violations, "Blocking calls found in AGO control plane/runtime:\n" + "\n".join(violations)


def test_static_no_blocking_or_readback_in_select_ago_active_set_gpu_async():
    """select_ago_active_set_gpu_async must have ZERO HOST_GUARD, vkMapMemory, or blocking waits."""
    trainer_cpp = REPO_ROOT / "vksplat" / "src" / "gs_trainer.cpp"
    body = _strip_cpp_comments(
        _extract_cpp_function_body(
            trainer_cpp.read_text(encoding="utf-8"),
            "void VulkanGSTrainer::select_ago_active_set_gpu_async(",
        )
    )
    banned = BANNED_CONTROL_PLANE_PATTERNS + [r"\bvkMapMemory\b"]
    violations = [pat for pat in banned if re.search(pat, body)]
    assert not violations, f"select_ago_active_set_gpu_async matched banned patterns: {violations}"


# ---------------------------------------------------------------------------
# Helpers for Fault-Injection Tests
# ---------------------------------------------------------------------------


def _make_action(
    action_id: int,
    region_id: int = 1,
    service_class=vksplat.AgoServiceClass.P2_REFINEMENT,
    op=vksplat.AgoOperationType.REFINE,
    expected_gain: float = 1.0,
    estimated_cost_ms: float = 0.5,
    req_map_ver: int = 0,
    req_pose_ver: int = 0,
    req_ev_ver: int = 0,
    exclusive: bool = False,
) -> vksplat.AgoAction:
    act = vksplat.AgoAction()
    act.action_id = action_id
    act.region_id = region_id
    act.service_class = service_class
    act.op = op
    act.expected_gain = expected_gain
    act.estimated_cost_ms = estimated_cost_ms
    act.required_map_version = req_map_ver
    act.required_pose_version = req_pose_ver
    act.required_evidence_version = req_ev_ver
    act.requires_exclusive_region = exclusive
    act.requires_gpu = True
    return act


def _make_budget(avail_ms: float = 10.0) -> vksplat.AgoDeadlineBudget:
    b = vksplat.AgoDeadlineBudget()
    b.now_ms = 0.0
    b.next_deadline_ms = avail_ms + 0.5
    b.reserved_ms = 0.0
    b.inflight_ms = 0.0
    b.safety_margin_ms = 0.5
    return b


# ---------------------------------------------------------------------------
# (b) 11 Fault-Injection Tests against Native C++ AgoScheduler
# ---------------------------------------------------------------------------


def test_fi01_slow_stalled_gpu_no_credits():
    """FI-1: Stalled GPU saturates 4-slot ring; scheduler rejects cleanly with NO_CREDITS and never blocks."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 4
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    exec_sim.set_stalled(True)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    for i in range(1, 7):
        assert sched.enqueue_action(_make_action(i, region_id=i, estimated_cost_ms=0.5))

    r1 = sched.tick()
    assert r1.actions_admitted == 4
    assert exec_sim.available_credits() == 0

    r2 = sched.tick()
    assert r2.actions_admitted == 0
    assert r2.credit_rejections >= 1

    # Unstall GPU and advance 1 tick -> completions free credits -> remaining 2 actions admitted.
    exec_sim.set_stalled(False)
    exec_sim.advance_ticks(1)
    r3 = sched.tick()
    assert r3.completions_processed == 4
    assert r3.actions_admitted == 2


def test_fi02_burst_telemetry_flood_drop_oldest():
    """FI-2: Telemetry flood drops oldest entries without blocking and never affects critical queue."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.telemetry_queue_capacity = 16
    cfg.critical_queue_capacity = 16
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)

    dropped_any = False
    for i in range(200):
        msg = vksplat.AgoEvidenceMessage()
        msg.kind = vksplat.AgoEvidenceKind.FRAME_OBSERVATION
        msg.region_id = (i % 8) + 1
        msg.magnitude = float(i)
        if sched.push_telemetry_evidence(msg):
            dropped_any = True
    assert dropped_any

    # Critical message still succeeds cleanly.
    crit = vksplat.AgoEvidenceMessage()
    crit.kind = vksplat.AgoEvidenceKind.CONTRADICTION
    crit.region_id = 42
    crit.critical = True
    assert sched.push_critical_evidence(crit)

    res = sched.tick()
    assert res.critical_evidence_processed == 1
    assert res.telemetry_evidence_processed == 16
    summary = sched.latest_summary()
    assert summary.total_dropped_telemetry == 200 - 16


def test_fi03_critical_queue_backpressure_never_drops():
    """FI-3: ReliableVersionedQueue returns False backpressure when full and NEVER drops accepted messages."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.critical_queue_capacity = 8
    cfg.max_critical_evidence_per_tick = 32
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)

    accepted = 0
    rejected = 0
    for i in range(1, 20):
        msg = vksplat.AgoEvidenceMessage()
        msg.kind = vksplat.AgoEvidenceKind.POSE_REVISION
        msg.region_id = i
        msg.pose_version = i + 1
        msg.critical = True
        if sched.push_critical_evidence(msg):
            accepted += 1
        else:
            rejected += 1

    assert accepted == 8
    assert rejected == 11

    res = sched.tick()
    assert res.critical_evidence_processed == 8
    assert sched.pose_version() == 9
    assert sched.latest_summary().total_critical_backpressure == 11


def test_fi04_stale_action_rejected_on_pose_revision():
    """FI-4: Pose revision bumps pose_version and prunes incompatible pending actions as REJECTED_STALE."""
    cfg = vksplat.AgoSchedulerConfig()
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    assert sched.enqueue_action(_make_action(1, region_id=1, req_pose_ver=1))
    assert sched.enqueue_action(_make_action(2, region_id=2, req_pose_ver=2))

    rev = vksplat.AgoEvidenceMessage()
    rev.kind = vksplat.AgoEvidenceKind.POSE_REVISION
    rev.region_id = 1
    rev.pose_version = 2
    rev.critical = True
    assert sched.push_critical_evidence(rev)

    res = sched.tick()
    assert sched.pose_version() == 2
    assert res.stale_actions_pruned == 1
    assert res.actions_admitted == 1
    assert exec_sim.submitted_order() == [2]
    assert sched.latest_summary().total_rejected_stale == 1


def test_fi05_stale_action_rejected_on_map_version_mismatch():
    """FI-5: Structural completion bumps map_version and prunes stale actions targeting old map_version."""
    cfg = vksplat.AgoSchedulerConfig()
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    assert sched.enqueue_action(_make_action(10, region_id=1, req_map_ver=1))
    assert sched.enqueue_action(_make_action(11, region_id=2, req_map_ver=2))

    comp = vksplat.AgoEvidenceMessage()
    comp.kind = vksplat.AgoEvidenceKind.STRUCTURAL_COMPLETION
    comp.region_id = 1
    comp.map_version = 2
    comp.critical = True
    assert sched.push_critical_evidence(comp)

    res = sched.tick()
    assert sched.map_version() == 2
    assert res.stale_actions_pruned == 1
    assert exec_sim.submitted_order() == [11]


def test_fi06_contradiction_p0_preempts_lower_classes():
    """FI-6: P0_CONSISTENCY always preempts P2_REFINEMENT and P3_PREDICTIVE regardless of raw gain."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 1
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    assert sched.enqueue_action(
        _make_action(1, region_id=1, service_class=vksplat.AgoServiceClass.P2_REFINEMENT, expected_gain=100.0)
    )
    assert sched.enqueue_action(
        _make_action(2, region_id=2, service_class=vksplat.AgoServiceClass.P3_PREDICTIVE, expected_gain=500.0)
    )
    assert sched.enqueue_action(
        _make_action(
            3,
            region_id=3,
            service_class=vksplat.AgoServiceClass.P0_CONSISTENCY,
            op=vksplat.AgoOperationType.RETRACT,
            expected_gain=0.1,
        )
    )

    res = sched.tick()
    assert res.actions_admitted == 1
    assert exec_sim.submitted_order() == [3]


def test_fi07_tight_or_zero_deadline_budget():
    """FI-7: Stage 2 strictly enforces B_t = max(0, d_next - t - C_reserved - C_inflight - m_t)."""
    cfg = vksplat.AgoSchedulerConfig()
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)

    # Zero available budget: nothing admitted
    sched.publish_deadline_budget(_make_budget(0.0))
    assert sched.enqueue_action(_make_action(1, region_id=1, estimated_cost_ms=2.0, expected_gain=10.0))
    assert sched.enqueue_action(_make_action(2, region_id=2, estimated_cost_ms=0.5, expected_gain=1.0))

    r1 = sched.tick()
    assert r1.actions_admitted == 0
    assert r1.budget_rejections >= 2

    # 1.0 ms budget: only the 0.5 ms action is admitted
    sched.publish_deadline_budget(_make_budget(1.0))
    r2 = sched.tick()
    assert r2.actions_admitted == 1
    assert exec_sim.submitted_order() == [2]


def test_fi08_marginal_utility_and_deterministic_tiebreak():
    """FI-8: Stage 3 selects argmax(expected_gain / estimated_cost_ms) with deterministic action_id tiebreak."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 4
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=2)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    # Utilities: id=10 -> 2.0, id=20 -> 5.0, id=5 -> 5.0 (tie with 20, lower id wins), id=30 -> 1.0
    assert sched.enqueue_action(_make_action(10, region_id=1, expected_gain=2.0, estimated_cost_ms=1.0))
    assert sched.enqueue_action(_make_action(20, region_id=2, expected_gain=5.0, estimated_cost_ms=1.0))
    assert sched.enqueue_action(_make_action(5, region_id=3, expected_gain=2.5, estimated_cost_ms=0.5))
    assert sched.enqueue_action(_make_action(30, region_id=4, expected_gain=1.0, estimated_cost_ms=1.0))

    res = sched.tick()
    assert res.actions_admitted == 4
    assert exec_sim.submitted_order() == [5, 20, 10, 30]


def test_fi09_p4_maintenance_floor_anti_starvation():
    """FI-9: P4_MAINTENANCE is guaranteed admission after maintenance_floor_interval_ticks under P2 load."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 1
    cfg.admission.maintenance_floor_interval_ticks = 3
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    assert sched.enqueue_action(
        _make_action(99, region_id=9, service_class=vksplat.AgoServiceClass.P4_MAINTENANCE, op=vksplat.AgoOperationType.SERIALIZE, expected_gain=0.1)
    )
    for i in range(1, 6):
        assert sched.enqueue_action(
            _make_action(i, region_id=i, service_class=vksplat.AgoServiceClass.P2_REFINEMENT, expected_gain=50.0)
        )

    for _ in range(4):
        sched.tick()
        exec_sim.advance_ticks(1)

    order = exec_sim.submitted_order()
    assert 99 in order[:3], f"Expected P4 maintenance floor to admit action 99 within 3 ticks, got {order}"


def test_fi10_frozen_retracted_and_exclusive_region_gating():
    """FI-10: Stage 1 blocks conflicting ops on frozen/retracted/exclusive regions and allows legal transitions."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 4
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    sched.set_region_frozen(10, True)
    sched.set_region_retracted(20, True)

    # Blocked on frozen/retracted regions
    assert sched.enqueue_action(_make_action(1, region_id=10, op=vksplat.AgoOperationType.REFINE))
    assert sched.enqueue_action(_make_action(2, region_id=20, op=vksplat.AgoOperationType.REFINE))
    # Allowed transitions
    assert sched.enqueue_action(_make_action(3, region_id=10, op=vksplat.AgoOperationType.REOPEN))
    assert sched.enqueue_action(_make_action(4, region_id=20, op=vksplat.AgoOperationType.RESTORE))

    # Exclusive region lock test on region 30
    assert sched.enqueue_action(_make_action(5, region_id=30, op=vksplat.AgoOperationType.SPLIT, exclusive=True))
    assert sched.enqueue_action(_make_action(6, region_id=30, op=vksplat.AgoOperationType.REFINE))

    r1 = sched.tick()
    assert r1.actions_admitted == 3
    assert set(exec_sim.submitted_order()) == {3, 4, 5}

    # Complete tick 1 actions -> region 10 reopened, region 20 restored, region 30 unlocked!
    exec_sim.advance_ticks(1)
    r2 = sched.tick()
    assert r2.completions_processed == 3
    assert r2.actions_admitted == 3
    assert set(exec_sim.submitted_order()[3:]) == {1, 2, 6}


def test_fi11_unsupported_op_and_execution_failure_recovery():
    """FI-11: Unsupported ops and execution failures release locks cleanly and keep EventTrace deterministic."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 2
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    # Mark MERGE as unsupported
    exec_sim.set_unsupported_op_mask(1 << int(vksplat.AgoOperationType.MERGE))
    exec_sim.set_fail_next_completion(True)

    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(20.0))

    assert sched.enqueue_action(
        _make_action(1, region_id=7, op=vksplat.AgoOperationType.MERGE, expected_gain=100.0)
    )
    assert sched.enqueue_action(
        _make_action(2, region_id=7, op=vksplat.AgoOperationType.SPLIT, expected_gain=10.0, exclusive=True)
    )
    assert sched.enqueue_action(
        _make_action(3, region_id=7, op=vksplat.AgoOperationType.REFINE, expected_gain=1.0)
    )

    r1 = sched.tick()
    # Action 1 rejected as unsupported; action 2 submitted exclusively (blocks action 3)
    assert sched.latest_summary().total_rejected_unsupported == 1
    assert r1.actions_admitted == 1
    assert exec_sim.submitted_order() == [2]

    # Action 2 completes with FAILED_EXECUTION -> must still release exclusive lock so action 3 runs!
    exec_sim.advance_ticks(1)
    r2 = sched.tick()
    assert r2.completions_processed == 1
    assert sched.latest_summary().total_failed == 1
    assert r2.actions_admitted == 1
    assert exec_sim.submitted_order() == [2, 3]

    # Non-blocking summary mailbox and deterministic trace populated
    consumed = sched.try_consume_summary()
    assert consumed is not None
    assert consumed.tick_count == 2
    trace = sched.trace_snapshot()
    event_types = [e.event_type for e in trace]
    assert vksplat.AgoTraceEventType.ACTION_REJECTED_UNSUPPORTED in event_types
    assert vksplat.AgoTraceEventType.COMPLETION_IN in event_types


# ---------------------------------------------------------------------------
# (c) CPU tick() Latency Measurement (p50 / p95 / p99 <= 100 us p95)
# ---------------------------------------------------------------------------


def test_cpu_tick_latency_under_100us_p95():
    """Measure native C++ AgoScheduler::tick() latency across 2000 ticks; assert p95 <= 100 us."""
    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 4
    cfg.max_pending_actions = 256
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)
    sched.publish_deadline_budget(_make_budget(16.0))

    # Warmup
    for i in range(100):
        sched.enqueue_action(_make_action(i + 1, region_id=(i % 32) + 1, expected_gain=float((i % 7) + 1)))
        sched.tick()
        exec_sim.advance_ticks(1)

    n_ticks = 2000
    cpp_tick_us = np.empty(n_ticks, dtype=np.float64)
    wall_tick_us = np.empty(n_ticks, dtype=np.float64)

    for i in range(n_ticks):
        ev = vksplat.AgoEvidenceMessage()
        ev.kind = vksplat.AgoEvidenceKind.FRAME_OBSERVATION
        ev.region_id = (i % 64) + 1
        ev.magnitude = 0.5
        sched.push_telemetry_evidence(ev)
        sched.enqueue_action(
            _make_action(1000 + i, region_id=(i % 64) + 1, expected_gain=float((i % 11) + 1), estimated_cost_ms=0.5)
        )
        t0 = time.perf_counter_ns()
        res = sched.tick()
        t1 = time.perf_counter_ns()
        cpp_tick_us[i] = res.tick_duration_us
        wall_tick_us[i] = (t1 - t0) * 1e-3
        exec_sim.advance_ticks(1)

    p50, p95, p99 = np.percentile(cpp_tick_us, [50, 95, 99])
    wp50, wp95, wp99 = np.percentile(wall_tick_us, [50, 95, 99])
    print(
        f"\n[AgoScheduler::tick()] C++ p50={p50:.2f}us p95={p95:.2f}us p99={p99:.2f}us | "
        f"PyBind wall p50={wp50:.2f}us p95={wp95:.2f}us p99={wp99:.2f}us"
    )
    assert p95 <= 100.0, f"AgoScheduler::tick() C++ p95={p95:.2f}us exceeded 100us budget"
    assert wp95 <= 100.0, f"AgoScheduler::tick() Python wall p95={wp95:.2f}us exceeded 100us budget"


# ---------------------------------------------------------------------------
# (d) GPU Parity Test (Run Serially by Coordinator with RUN_GPU_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU parity test runs serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_async_cutoff_parity_and_empty_state(tmp_path):
    """
    GPU parity between legacy select_ago_exact_k_gpu (S0) and non-blocking
    select_ago_active_set_gpu_async (P2), plus explicit EMPTY -> groupCountX == 0
    and AGO-0 external_strategy_mode checks.
    """
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_async_ago", n_points=1024, w=64, h=48, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_async_ago", mcmc_protect_structural_groups=True)
    vk.force_backward_variant(1)
    vk.set_ago_telemetry_enabled(True)
    vk.reset_ago_telemetry()
    for s in range(1, 13):
        vk.train_step(s % int(vk.num_train), s)

    n = int(len(vk.xyz_ws))
    assert n == 1024

    # 1. Partial parity (target_fraction = 0.5)
    s0 = vk.select_ago_exact_k_gpu(target_fraction=0.5, newborn_warmup_steps=10.0, exclude_frozen=True)
    vk.select_ago_active_set_gpu_async(target_fraction=0.5, newborn_warmup_steps=10.0, exclude_frozen=True)
    async_st = vk.get_ago_cutoff_indirect_state()
    assert async_st["active_set_state"] == int(vksplat.AgoActiveSetState.PARTIAL)
    assert async_st["selected_count"] == s0["active_count"]
    assert async_st["group_count_x"] > 0

    # 2. Full parity (target_fraction = 1.0)
    vk.select_ago_active_set_gpu_async(target_fraction=1.0, newborn_warmup_steps=10.0, exclude_frozen=True)
    full_st = vk.get_ago_cutoff_indirect_state()
    assert full_st["active_set_state"] == int(vksplat.AgoActiveSetState.FULL)
    assert full_st["selected_count"] == n
    assert full_st["group_count_x"] > 0

    # 3. Empty active set (target_fraction = 0.0) MUST encode EMPTY and groupCountX == 0 (never fallback to full)
    vk.select_ago_active_set_gpu_async(target_fraction=0.0, newborn_warmup_steps=10.0, exclude_frozen=True)
    empty_st = vk.get_ago_cutoff_indirect_state()
    assert empty_st["active_set_state"] == int(vksplat.AgoActiveSetState.EMPTY)
    assert empty_st["selected_count"] == 0
    assert empty_st["group_count_x"] == 0

    # 4. AGO-0 external_strategy_mode explicit unsupported-op error
    vk.set_external_strategy_mode(True)
    assert vk.get_external_strategy_mode() is True
    st_unsup = vk.ago_execute_external_op(vksplat.AgoOperationType.MERGE, -1)
    assert st_unsup == vksplat.AgoCompletionStatus.REJECTED_UNSUPPORTED

    # 5. Latency comparison (0-readback GPU-native cutoff vs S0 2064B readback)
    t0 = time.perf_counter()
    for _ in range(50):
        vk.select_ago_active_set_gpu_async(target_fraction=0.5, newborn_warmup_steps=10.0, exclude_frozen=True)
    t_async_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    print(f"\n[P2 GPU-Native Zero-Readback Cutoff N={n}] avg={t_async_ms:.4f} ms/call (0 bytes host readback)")
    assert t_async_ms < 0.5
