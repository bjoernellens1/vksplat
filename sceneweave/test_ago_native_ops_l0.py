"""
Phase 10 W1-W3 Verification Suite:
  (a) Static invariants banning spin-locks in mailbox.h and null-cb acceptance in gpu_executor.cpp,
      plus CPU runtime checks for REJECTED_NO_EXECUTOR and event_to_map_correction_ms.
  (b) GPU-marked AGO-L0 invariants (@pytest.mark.gpu, run serially by Coordinator with RUN_GPU_TESTS=1):
      - Freeze / Reopen / Retract / Prune GPU device sync & render disappearance
      - PARAM_SH-only bit-exactness on xyz/rot/scales/opac
      - RAM offload / restore bit-exactness (params + Adam moments + group states)
      - Real NonBlockingGpuExecutor submission/completion + event_to_map_correction_ms
  (c) Rust cdylib 4-function C ABI parity check via ctypes + Rust-vs-C++ trace replay microbenchmark.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "vksplat" / "build"
if BUILD_DIR.exists() and str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import vksplat  # noqa: E402

RUST_CDYLIB = (
    REPO_ROOT
    / "vksplat"
    / "ago_control_plane"
    / "target"
    / "release"
    / "libago_control_plane.so"
)


def _strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


# ---------------------------------------------------------------------------
# (a) Static & CPU Runtime Hardening Checks
# ---------------------------------------------------------------------------


def test_static_no_spinlock_in_mailbox_h():
    """MANDATORY P0: Zero test_and_set or atomic_flag spin-locks in mailbox.h."""
    mailbox_h = REPO_ROOT / "vksplat" / "src" / "ago" / "core" / "mailbox.h"
    code = _strip_cpp_comments(mailbox_h.read_text(encoding="utf-8"))
    assert not re.search(r"\btest_and_set\b", code), "Found banned test_and_set spin-lock in mailbox.h"
    assert not re.search(r"\batomic_flag\b", code), "Found banned atomic_flag spin-lock in mailbox.h"


def test_static_and_runtime_null_cb_rejected_in_gpu_executor():
    """MANDATORY P0: NonBlockingGpuExecutor rejects null record_cb_ and zero-work submissions."""
    gpu_exec_cpp = REPO_ROOT / "vksplat" / "src" / "ago" / "runtime" / "gpu_executor.cpp"
    code = _strip_cpp_comments(gpu_exec_cpp.read_text(encoding="utf-8"))
    assert "REJECTED_NO_EXECUTOR" in code, "gpu_executor.cpp must return SubmitResult::REJECTED_NO_EXECUTOR"
    assert re.search(r"if\s*\(\s*!record_cb_\s*\)", code), "gpu_executor.cpp must check !record_cb_"
    assert re.search(r"recorded_ops\s*==\s*0", code), "gpu_executor.cpp must reject zero recorded work"

    # Runtime CPU check: unconfigured executor (null record_cb_) MUST return REJECTED_NO_EXECUTOR, never OK.
    exec_gpu = vksplat.NonBlockingGpuExecutor()
    act = vksplat.AgoAction()
    act.action_id = 1
    act.region_id = 1
    act.op = vksplat.AgoOperationType.REFINE
    act.estimated_cost_ms = 0.5
    act.expected_gain = 1.0
    assert exec_gpu.try_submit(act) == vksplat.AgoSubmitResult.REJECTED_NO_EXECUTOR


def test_cpu_event_to_map_correction_ms_tracking():
    """Verify AgoScheduler tracks event_to_map_correction_ms from evidence ingestion to completion."""
    cfg = vksplat.AgoSchedulerConfig()
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)

    budget = vksplat.AgoDeadlineBudget()
    budget.now_ms = 0.0
    budget.next_deadline_ms = 16.5
    budget.safety_margin_ms = 0.5
    sched.publish_deadline_budget(budget)

    ev = vksplat.AgoEvidenceMessage()
    ev.kind = vksplat.AgoEvidenceKind.CONTRADICTION
    ev.region_id = 42
    ev.magnitude = 1.5
    ev.critical = True
    assert sched.push_critical_evidence(ev)
    sched.tick()

    act = vksplat.AgoAction()
    act.action_id = 7
    act.region_id = 42
    act.service_class = vksplat.AgoServiceClass.P0_CONSISTENCY
    act.op = vksplat.AgoOperationType.RETRACT
    act.expected_gain = 2.0
    act.estimated_cost_ms = 0.5
    assert sched.enqueue_action(act)

    r1 = sched.tick()
    assert r1.actions_admitted == 1
    time.sleep(0.002)
    exec_sim.advance_ticks(1)
    r2 = sched.tick()
    assert r2.completions_processed == 1

    summary = sched.latest_summary()
    assert summary.event_to_map_correction_samples == 1
    assert summary.last_event_to_map_correction_ms >= 1.5
    assert summary.max_event_to_map_correction_ms >= summary.last_event_to_map_correction_ms
    assert summary.mean_event_to_map_correction_ms > 0.0


# ---------------------------------------------------------------------------
# (c) Rust cdylib C ABI Parity Check via ctypes + Rust-vs-C++ Microbenchmark
# ---------------------------------------------------------------------------


class _AgoCBudget(ctypes.Structure):
    _fields_ = [
        ("now_ms", ctypes.c_double),
        ("next_deadline_ms", ctypes.c_double),
        ("reserved_ms", ctypes.c_double),
        ("inflight_ms", ctypes.c_double),
        ("safety_margin_ms", ctypes.c_double),
    ]


class _AgoCEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint8),
        ("critical", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 2),
        ("flags", ctypes.c_uint32),
        ("region_id", ctypes.c_uint64),
        ("map_version", ctypes.c_uint64),
        ("pose_version", ctypes.c_uint64),
        ("evidence_version", ctypes.c_uint64),
        ("timestamp_ms", ctypes.c_double),
        ("magnitude", ctypes.c_float),
        ("_pad1", ctypes.c_uint32),
    ]


class _AgoCAction(ctypes.Structure):
    _fields_ = [
        ("action_id", ctypes.c_uint64),
        ("region_id", ctypes.c_uint64),
        ("service_class", ctypes.c_uint8),
        ("op", ctypes.c_uint8),
        ("requires_exclusive_region", ctypes.c_uint8),
        ("requires_gpu", ctypes.c_uint8),
        ("max_splats", ctypes.c_uint32),
        ("required_map_version", ctypes.c_uint64),
        ("required_pose_version", ctypes.c_uint64),
        ("required_evidence_version", ctypes.c_uint64),
        ("expected_gain", ctypes.c_float),
        ("estimated_cost_ms", ctypes.c_float),
        ("target_fraction", ctypes.c_float),
        ("_pad0", ctypes.c_uint32),
        ("enqueued_at_ms", ctypes.c_double),
        ("evidence_ingested_at_ms", ctypes.c_double),
        ("deadline_ms", ctypes.c_double),
    ]


class _AgoCCompletion(ctypes.Structure):
    _fields_ = [
        ("action_id", ctypes.c_uint64),
        ("region_id", ctypes.c_uint64),
        ("op", ctypes.c_uint8),
        ("service_class", ctypes.c_uint8),
        ("status", ctypes.c_uint8),
        ("was_exclusive", ctypes.c_uint8),
        ("affected_splats", ctypes.c_uint32),
        ("realized_gain", ctypes.c_float),
        ("actual_cost_ms", ctypes.c_float),
        ("completed_at_ms", ctypes.c_double),
        ("event_to_map_correction_ms", ctypes.c_double),
        ("result_map_version", ctypes.c_uint64),
        ("result_pose_version", ctypes.c_uint64),
    ]


class _AgoCResourceSummary(ctypes.Structure):
    _fields_ = [
        ("tick_count", ctypes.c_uint64),
        ("map_version", ctypes.c_uint64),
        ("pose_version", ctypes.c_uint64),
        ("evidence_version", ctypes.c_uint64),
        ("inflight_count", ctypes.c_uint32),
        ("queued_actions", ctypes.c_uint32),
        ("dirty_regions", ctypes.c_uint32),
        ("available_credits", ctypes.c_uint32),
        ("total_admitted", ctypes.c_uint64),
        ("total_completed", ctypes.c_uint64),
        ("total_rejected_stale", ctypes.c_uint64),
        ("total_rejected_budget", ctypes.c_uint64),
        ("total_rejected_credits", ctypes.c_uint64),
        ("total_failed", ctypes.c_uint64),
        ("available_budget_ms", ctypes.c_double),
        ("last_event_to_map_correction_ms", ctypes.c_double),
        ("max_event_to_map_correction_ms", ctypes.c_double),
        ("mean_event_to_map_correction_ms", ctypes.c_double),
    ]


def _load_rust_cdylib() -> ctypes.CDLL:
    assert RUST_CDYLIB.exists(), f"Rust cdylib not built: {RUST_CDYLIB}"
    lib = ctypes.CDLL(str(RUST_CDYLIB))
    lib.ago_cp_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.ago_cp_create.restype = ctypes.c_void_p
    lib.ago_cp_destroy.argtypes = [ctypes.c_void_p]
    lib.ago_cp_destroy.restype = None
    lib.ago_cp_set_budget.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AgoCBudget)]
    lib.ago_cp_set_budget.restype = ctypes.c_int32
    lib.ago_cp_enqueue_actions.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AgoCAction), ctypes.c_size_t]
    lib.ago_cp_enqueue_actions.restype = ctypes.c_uint32
    lib.ago_ingest_events.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AgoCEvent), ctypes.c_size_t]
    lib.ago_ingest_events.restype = ctypes.c_uint32
    lib.ago_schedule_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AgoCAction), ctypes.c_size_t]
    lib.ago_schedule_batch.restype = ctypes.c_uint32
    lib.ago_ingest_completions.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AgoCCompletion), ctypes.c_size_t]
    lib.ago_ingest_completions.restype = ctypes.c_uint32
    lib.ago_get_resource_summary.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AgoCResourceSummary)]
    lib.ago_get_resource_summary.restype = ctypes.c_int32
    return lib


def test_rust_cdylib_c_abi_parity_with_cpp_scheduler():
    """Verify exact decision parity between Rust cdylib C ABI and C++ AgoScheduler on identical trace."""
    lib = _load_rust_cdylib()
    cp = lib.ago_cp_create(4, 8)
    assert cp

    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 1
    cfg.admission.maintenance_floor_interval_ticks = 8
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)

    try:
        # Identical budget: 10.0 ms available
        rb = _AgoCBudget(0.0, 10.5, 0.0, 0.0, 0.5)
        assert lib.ago_cp_set_budget(cp, ctypes.byref(rb)) == 0
        cb = vksplat.AgoDeadlineBudget()
        cb.now_ms = 0.0
        cb.next_deadline_ms = 10.5
        cb.safety_margin_ms = 0.5
        sched.publish_deadline_budget(cb)

        # Step 1: Pose revision bumps pose_version 1 -> 2
        rev_r = _AgoCEvent(
            kind=int(vksplat.AgoEvidenceKind.POSE_REVISION),
            critical=1,
            region_id=7,
            pose_version=2,
            timestamp_ms=10.0,
            magnitude=1.0,
        )
        assert lib.ago_ingest_events(cp, ctypes.byref(rev_r), 1) == 1

        rev_c = vksplat.AgoEvidenceMessage()
        rev_c.kind = vksplat.AgoEvidenceKind.POSE_REVISION
        rev_c.critical = True
        rev_c.region_id = 7
        rev_c.pose_version = 2
        rev_c.timestamp_ms = 10.0
        rev_c.magnitude = 1.0
        assert sched.push_critical_evidence(rev_c)

        # Enqueue 3 actions: stale (pose_ver=1), P2 high-gain, P0 consistency
        spec = [
            (1, 7, vksplat.AgoServiceClass.P2_REFINEMENT, vksplat.AgoOperationType.REFINE, 1, 500.0, 0.5),
            (2, 8, vksplat.AgoServiceClass.P2_REFINEMENT, vksplat.AgoOperationType.REFINE, 2, 100.0, 0.5),
            (3, 7, vksplat.AgoServiceClass.P0_CONSISTENCY, vksplat.AgoOperationType.RETRACT, 2, 0.5, 0.5),
        ]
        r_acts = (_AgoCAction * len(spec))()
        for idx, (aid, rid, sc_cls, op, pver, gain, cost) in enumerate(spec):
            r_acts[idx] = _AgoCAction(
                action_id=aid,
                region_id=rid,
                service_class=int(sc_cls),
                op=int(op),
                requires_gpu=1,
                required_pose_version=pver,
                expected_gain=gain,
                estimated_cost_ms=cost,
            )
            ca = vksplat.AgoAction()
            ca.action_id = aid
            ca.region_id = rid
            ca.service_class = sc_cls
            ca.op = op
            ca.requires_gpu = True
            ca.required_pose_version = pver
            ca.expected_gain = gain
            ca.estimated_cost_ms = cost
            assert sched.enqueue_action(ca)
        assert lib.ago_cp_enqueue_actions(cp, r_acts, len(spec)) == len(spec)

        # Tick 1: both must prune #1 as stale and admit P0 #3 first
        r_out = (_AgoCAction * 1)()
        assert lib.ago_schedule_batch(cp, r_out, 1) == 1
        c_t1 = sched.tick()
        assert c_t1.actions_admitted == 1
        assert r_out[0].action_id == exec_sim.submitted_order()[0] == 3

        # Complete #3 in both
        comp_r = _AgoCCompletion(
            action_id=3,
            region_id=7,
            op=int(vksplat.AgoOperationType.RETRACT),
            service_class=int(vksplat.AgoServiceClass.P0_CONSISTENCY),
            status=int(vksplat.AgoCompletionStatus.OK),
            actual_cost_ms=0.5,
            completed_at_ms=13.25,
        )
        assert lib.ago_ingest_completions(cp, ctypes.byref(comp_r), 1) == 1
        exec_sim.advance_ticks(1)

        # Tick 2: both admit #2
        assert lib.ago_schedule_batch(cp, r_out, 1) == 1
        c_t2 = sched.tick()
        assert c_t2.actions_admitted == 1
        assert r_out[0].action_id == exec_sim.submitted_order()[1] == 2

        r_sum = _AgoCResourceSummary()
        assert lib.ago_get_resource_summary(cp, ctypes.byref(r_sum)) == 0
        c_sum = sched.latest_summary()

        assert r_sum.pose_version == c_sum.pose_version == 2
        assert r_sum.map_version == c_sum.map_version == 2
        assert r_sum.total_rejected_stale == c_sum.total_rejected_stale == 1
        assert r_sum.total_admitted == c_sum.total_admitted == 2
        assert r_sum.total_completed == c_sum.total_completed == 1
        assert abs(r_sum.last_event_to_map_correction_ms - 3.25) < 1e-6
    finally:
        lib.ago_cp_destroy(cp)


def test_rust_vs_cpp_trace_replay_microbenchmark():
    """Compare Rust cdylib C ABI vs C++ AgoScheduler p50/p95/p99 tick latency on identical 1000-tick trace."""
    lib = _load_rust_cdylib()
    cp = lib.ago_cp_create(4, 8)
    assert cp

    cfg = vksplat.AgoSchedulerConfig()
    cfg.max_admissions_per_tick = 4
    exec_sim = vksplat.SimulatedAgoExecutor(ring_slots=4, default_latency_ticks=1)
    sched = vksplat.AgoScheduler(cfg, exec_sim)

    try:
        rb = _AgoCBudget(0.0, 16.5, 0.0, 0.0, 0.5)
        lib.ago_cp_set_budget(cp, ctypes.byref(rb))
        cb = vksplat.AgoDeadlineBudget()
        cb.now_ms = 0.0
        cb.next_deadline_ms = 16.5
        cb.safety_margin_ms = 0.5
        sched.publish_deadline_budget(cb)

        n_ticks = 1000
        cpp_us = np.empty(n_ticks, dtype=np.float64)
        rust_us = np.empty(n_ticks, dtype=np.float64)

        r_out = (_AgoCAction * 4)()
        r_comp = (_AgoCCompletion * 4)()

        for i in range(n_ticks):
            rid = (i % 64) + 1
            gain = float((i % 11) + 1)

            # C++ tick
            ev_c = vksplat.AgoEvidenceMessage()
            ev_c.kind = vksplat.AgoEvidenceKind.FRAME_OBSERVATION
            ev_c.region_id = rid
            ev_c.magnitude = 0.5
            sched.push_telemetry_evidence(ev_c)
            act_c = vksplat.AgoAction()
            act_c.action_id = i + 1
            act_c.region_id = rid
            act_c.service_class = vksplat.AgoServiceClass.P2_REFINEMENT
            act_c.op = vksplat.AgoOperationType.REFINE
            act_c.expected_gain = gain
            act_c.estimated_cost_ms = 0.5
            sched.enqueue_action(act_c)
            res = sched.tick()
            cpp_us[i] = res.tick_duration_us
            exec_sim.advance_ticks(1)

            # Rust cdylib tick (measured across ctypes boundary)
            ev_r = _AgoCEvent(kind=0, region_id=rid, magnitude=0.5, timestamp_ms=float(i))
            act_r = _AgoCAction(
                action_id=i + 1,
                region_id=rid,
                service_class=2,
                op=0,
                requires_gpu=1,
                expected_gain=gain,
                estimated_cost_ms=0.5,
            )
            t0 = time.perf_counter_ns()
            lib.ago_ingest_events(cp, ctypes.byref(ev_r), 1)
            lib.ago_cp_enqueue_actions(cp, ctypes.byref(act_r), 1)
            n_adm = lib.ago_schedule_batch(cp, r_out, 4)
            for j in range(n_adm):
                r_comp[j] = _AgoCCompletion(
                    action_id=r_out[j].action_id,
                    region_id=r_out[j].region_id,
                    op=r_out[j].op,
                    service_class=r_out[j].service_class,
                    status=0,
                    actual_cost_ms=0.5,
                    completed_at_ms=float(i) + 0.5,
                )
            if n_adm > 0:
                lib.ago_ingest_completions(cp, r_comp, n_adm)
            rust_us[i] = (time.perf_counter_ns() - t0) * 1e-3

        cp50, cp95, cp99 = np.percentile(cpp_us, [50, 95, 99])
        rp50, rp95, rp99 = np.percentile(rust_us, [50, 95, 99])
        print(
            f"\n[Trace Replay N={n_ticks}] "
            f"C++ inner p50={cp50:.2f}us p95={cp95:.2f}us p99={cp99:.2f}us | "
            f"Rust ctypes p50={rp50:.2f}us p95={rp95:.2f}us p99={rp99:.2f}us"
        )
        assert cp95 <= 100.0
        assert rp95 <= 100.0
    finally:
        lib.ago_cp_destroy(cp)


# ---------------------------------------------------------------------------
# (b) GPU-Marked AGO-L0 Invariants (Coordinator runs serially with RUN_GPU_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU AGO-L0 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_l0_freeze_reopen_retract_prune(tmp_path):
    """Verify Freeze/Reopen/Retract/Prune sync directly to GPU and affect optimizer + renders."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_l0_ops", n_points=512, w=64, h=48, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_l0_ops", mcmc_protect_structural_groups=True)
    vk.force_backward_variant(1)
    vk.set_external_strategy_mode(True)

    n = int(len(vk.xyz_ws))
    half = n // 2
    gids = np.zeros(n, dtype=np.int32)
    gids[half:] = 1
    vk.set_splat_group_ids(gids)
    vk.set_group_state(0, 0)
    vk.set_group_state(1, 0)

    # Warmup 2 steps so all buffers/moments are initialized
    for s in range(1, 3):
        vk.train_step(s % int(vk.num_train), s)

    # 1. FREEZE group 0 -> group 0 params MUST stay bit-exact while group 1 updates
    assert vk.ago_execute_external_op(vksplat.AgoOperationType.FREEZE, 0) == vksplat.AgoCompletionStatus.OK
    xyz_before = np.array(vk.xyz_ws, copy=True)
    rot_before = np.array(vk.rotations, copy=True)
    sh_before = np.array(vk.sh_coeffs, copy=True)

    for s in range(3, 8):
        vk.train_step(s % int(vk.num_train), s)

    xyz_after = np.array(vk.xyz_ws, copy=True)
    rot_after = np.array(vk.rotations, copy=True)
    sh_after = np.array(vk.sh_coeffs, copy=True)
    assert np.array_equal(xyz_after[:half], xyz_before[:half]), "Frozen group 0 xyz_ws mutated!"
    assert np.array_equal(rot_after[:half], rot_before[:half]), "Frozen group 0 rotations mutated!"
    assert np.array_equal(sh_after[:half], sh_before[:half]), "Frozen group 0 sh_coeffs mutated!"
    assert not np.array_equal(xyz_after[half:], xyz_before[half:]), "Active group 1 did not update!"

    # 2. REOPEN group 0 -> group 0 resumes updating
    assert vk.ago_execute_external_op(vksplat.AgoOperationType.REOPEN, 0) == vksplat.AgoCompletionStatus.OK
    for s in range(8, 12):
        vk.train_step(s % int(vk.num_train), s)
    xyz_reopened = np.array(vk.xyz_ws, copy=True)
    assert not np.array_equal(xyz_reopened[:half], xyz_before[:half]), "Reopened group 0 did not update!"

    # 3. RETRACT group 0 -> opacities masked to 0 on GPU and render changes
    rgbd_before = vk.render_val_depth(0)
    img_before = np.array(rgbd_before["rgb"], copy=True)
    assert img_before.ndim >= 2 and img_before.size > 0
    assert vk.ago_execute_external_op(vksplat.AgoOperationType.RETRACT, 0) == vksplat.AgoCompletionStatus.OK
    opacs_after_retract = np.array(vk.opacities, copy=True)
    assert float(np.max(opacs_after_retract[:half])) == 0.0, "Retracted group 0 opacities must be 0 on GPU!"
    rgbd_after = vk.render_val_depth(0)
    img_after = np.array(rgbd_after["rgb"], copy=True)
    assert not np.array_equal(img_before, img_after), "Retracted splats did not disappear from GPU render!"

    # 4. PRUNE group 0 -> slots compacted and freed
    assert vk.ago_execute_external_op(vksplat.AgoOperationType.PRUNE, 0) == vksplat.AgoCompletionStatus.OK
    assert int(len(vk.xyz_ws)) == n - half


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU AGO-L0 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_l0_param_sh_only_bit_exact(tmp_path):
    """Verify PARAM_SH (16) updates ONLY SH while xyz/rot/scales/opac remain bit-exact."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_l0_pmask", n_points=512, w=64, h=48, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_l0_pmask", mcmc_protect_structural_groups=True)
    vk.force_backward_variant(1)
    vk.set_external_strategy_mode(True)

    for s in range(1, 4):
        vk.train_step(s % int(vk.num_train), s)

    vk.set_ago_param_mask(vksplat.AGO_PARAM_SH)
    assert vk.get_ago_param_mask() == 16

    xyz0 = np.array(vk.xyz_ws, copy=True)
    rot0 = np.array(vk.rotations, copy=True)
    scl0 = np.array(vk.scales, copy=True)
    opa0 = np.array(vk.opacities, copy=True)
    sh0 = np.array(vk.sh_coeffs, copy=True)

    for s in range(4, 9):
        vk.train_step(s % int(vk.num_train), s)

    assert np.array_equal(np.array(vk.xyz_ws), xyz0), "PARAM_SH mutated xyz_ws!"
    assert np.array_equal(np.array(vk.rotations), rot0), "PARAM_SH mutated rotations!"
    assert np.array_equal(np.array(vk.scales), scl0), "PARAM_SH mutated scales!"
    assert np.array_equal(np.array(vk.opacities), opa0), "PARAM_SH mutated opacities!"
    assert not np.array_equal(np.array(vk.sh_coeffs), sh0), "PARAM_SH did not update sh_coeffs!"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU AGO-L0 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_ago_l0_ram_offload_restore_bit_exact(tmp_path):
    """Verify ago_offload_snapshot_to_ram / ago_restore_snapshot_from_ram preserves params + Adam moments bit-exact."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_l0_ram", n_points=512, w=64, h=48, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_l0_ram", mcmc_protect_structural_groups=True)
    vk.force_backward_variant(1)
    vk.set_external_strategy_mode(True)

    for s in range(1, 6):
        vk.train_step(s % int(vk.num_train), s)

    snap_id = vk.ago_offload_snapshot_to_ram()
    assert snap_id > 0
    assert vk.ago_ram_snapshot_bytes(snap_id) > 0

    xyz_snap = np.array(vk.xyz_ws, copy=True)
    sh_snap = np.array(vk.sh_coeffs, copy=True)
    rot_snap = np.array(vk.rotations, copy=True)
    scl_snap = np.array(vk.scales, copy=True)
    opa_snap = np.array(vk.opacities, copy=True)

    # Step 6 reference (captures exact post-step state given current Adam moments)
    vk.train_step(0, 6)
    xyz_after_6_ref = np.array(vk.xyz_ws, copy=True)
    assert not np.array_equal(xyz_after_6_ref, xyz_snap)

    # Mutate further
    for s in range(7, 11):
        vk.train_step(s % int(vk.num_train), s)

    # Restore bit-exact snapshot
    assert vk.ago_restore_snapshot_from_ram(snap_id) is True
    assert np.array_equal(np.array(vk.xyz_ws), xyz_snap)
    assert np.array_equal(np.array(vk.sh_coeffs), sh_snap)
    assert np.array_equal(np.array(vk.rotations), rot_snap)
    assert np.array_equal(np.array(vk.scales), scl_snap)
    assert np.array_equal(np.array(vk.opacities), opa_snap)

    # Re-run step 6 -> must match reference bit-exact (proving Adam moments were restored!)
    vk.train_step(0, 6)
    assert np.array_equal(np.array(vk.xyz_ws), xyz_after_6_ref), "Adam moments diverged after RAM restore!"
    assert vk.ago_drop_ram_snapshot(snap_id) is True


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS", "0") != "1",
    reason="GPU AGO-L0 tests run serially under Coordinator control (set RUN_GPU_TESTS=1)",
)
def test_gpu_nonblocking_executor_submission_completion_and_e2m(tmp_path):
    """Verify real Vulkan NonBlockingGpuExecutor submission, fence completion, e2m tracking, and clean shutdown."""
    import append_scenario as sc

    ds = sc.build_dataset(tmp_path / "ds_l0_exec", n_points=256, w=64, h=48, structured=True)
    vk = sc.new_trainer(ds, tmp_path / "out_l0_exec", mcmc_protect_structural_groups=True)

    cfg = vksplat.AgoSchedulerConfig()
    vk.ago_scheduler_init(cfg, use_gpu_executor=True)
    budget = vksplat.AgoDeadlineBudget()
    budget.now_ms = 0.0
    budget.next_deadline_ms = 16.5
    budget.safety_margin_ms = 0.5
    vk.ago_scheduler_set_budget(budget)

    ev = vksplat.AgoEvidenceMessage()
    ev.kind = vksplat.AgoEvidenceKind.CONTRADICTION
    ev.region_id = 1
    ev.magnitude = 1.0
    ev.critical = True
    assert vk.ago_scheduler_push_evidence(ev)
    vk.ago_scheduler_tick()

    act = vksplat.AgoAction()
    act.action_id = 101
    act.region_id = 1
    act.service_class = vksplat.AgoServiceClass.P0_CONSISTENCY
    act.op = vksplat.AgoOperationType.RETRACT
    act.expected_gain = 1.0
    act.estimated_cost_ms = 0.5
    assert vk.ago_scheduler_enqueue_action(act)

    r1 = vk.ago_scheduler_tick()
    assert r1.actions_admitted == 1

    # Poll non-blocking until GPU fence signals
    completed = 0
    for _ in range(200):
        r = vk.ago_scheduler_tick()
        completed += r.completions_processed
        if completed >= 1:
            break
        time.sleep(0.001)

    assert completed == 1
    summary = vk.ago_scheduler_summary()
    assert summary.total_completed == 1
    assert summary.event_to_map_correction_samples == 1
    assert summary.last_event_to_map_correction_ms > 0.0

    vk.ago_gpu_shutdown()
