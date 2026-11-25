//! Hierarchical Rust AGO Control Plane with 4-function batched C ABI.

pub mod envelope;
pub mod state_machine;

use std::slice;

pub use envelope::{
    AgoCAction, AgoCBudget, AgoCCompletion, AgoCEvent, AgoCResourceSummary, ControlPlane,
};
pub use state_machine::{ActionState, TransitionError};

/// Create a new heap-allocated `ControlPlane` instance for C/Python callers.
#[no_mangle]
pub extern "C" fn ago_cp_create(max_inflight: u32, maintenance_floor_ticks: u32) -> *mut ControlPlane {
    Box::into_raw(Box::new(ControlPlane::new(max_inflight, maintenance_floor_ticks)))
}

/// Destroy a `ControlPlane` instance created by `ago_cp_create`.
#[no_mangle]
pub unsafe extern "C" fn ago_cp_destroy(handle: *mut ControlPlane) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Publish a deadline budget snapshot. Returns 0 on success, -1 on null pointer.
#[no_mangle]
pub unsafe extern "C" fn ago_cp_set_budget(
    handle: *mut ControlPlane,
    budget: *const AgoCBudget,
) -> i32 {
    if handle.is_null() || budget.is_null() {
        return -1;
    }
    (*handle).set_budget(*budget);
    0
}

/// Enqueue candidate actions into the pending pool. Returns count accepted.
#[no_mangle]
pub unsafe extern "C" fn ago_cp_enqueue_actions(
    handle: *mut ControlPlane,
    actions_ptr: *const AgoCAction,
    action_count: usize,
) -> u32 {
    if handle.is_null() || actions_ptr.is_null() || action_count == 0 {
        return 0;
    }
    let actions = slice::from_raw_parts(actions_ptr, action_count);
    (*handle).enqueue_actions(actions)
}

/// Core C ABI #1: Batch-ingest evidence events. Returns count processed.
#[no_mangle]
pub unsafe extern "C" fn ago_ingest_events(
    handle: *mut ControlPlane,
    events_ptr: *const AgoCEvent,
    event_count: usize,
) -> u32 {
    if handle.is_null() || events_ptr.is_null() || event_count == 0 {
        return 0;
    }
    let events = slice::from_raw_parts(events_ptr, event_count);
    (*handle).ingest_events(events)
}

/// Core C ABI #2: Run one bounded control-plane scheduling batch. Returns actions admitted.
#[no_mangle]
pub unsafe extern "C" fn ago_schedule_batch(
    handle: *mut ControlPlane,
    out_actions_ptr: *mut AgoCAction,
    max_actions: usize,
) -> u32 {
    if handle.is_null() || out_actions_ptr.is_null() || max_actions == 0 {
        return 0;
    }
    let out = slice::from_raw_parts_mut(out_actions_ptr, max_actions);
    (*handle).schedule_batch(out)
}

/// Core C ABI #3: Batch-ingest GPU completions and update `event_to_map_correction_ms`.
#[no_mangle]
pub unsafe extern "C" fn ago_ingest_completions(
    handle: *mut ControlPlane,
    completions_ptr: *const AgoCCompletion,
    completion_count: usize,
) -> u32 {
    if handle.is_null() || completions_ptr.is_null() || completion_count == 0 {
        return 0;
    }
    let completions = slice::from_raw_parts(completions_ptr, completion_count);
    (*handle).ingest_completions(completions)
}

/// Core C ABI #4: Read current resource/version/latency summary. Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn ago_get_resource_summary(
    handle: *const ControlPlane,
    out_summary_ptr: *mut AgoCResourceSummary,
) -> i32 {
    if handle.is_null() || out_summary_ptr.is_null() {
        return -1;
    }
    *out_summary_ptr = (*handle).resource_summary();
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{
        COMPLETION_OK, EVIDENCE_POSE_REVISION, OP_FREEZE, OP_REFINE, OP_REOPEN, OP_RETRACT,
        OP_SERIALIZE, SERVICE_P0_CONSISTENCY, SERVICE_P2_REFINEMENT, SERVICE_P4_MAINTENANCE,
    };

    #[test]
    fn action_state_typestate_transitions() {
        let s = ActionState::Candidate;
        // Cannot submit directly from Candidate without becoming Ready
        assert!(s.submit(1).is_err());
        assert!(s.complete().is_err());

        let s = s.mark_ready().unwrap();
        assert_eq!(s, ActionState::Ready);

        let s = s.submit(42).unwrap();
        assert_eq!(s, ActionState::Submitted(42));
        // Submitted cannot go back to Ready or re-submit
        assert!(s.mark_ready().is_err());
        assert!(s.submit(43).is_err());

        let s = s.complete().unwrap();
        assert_eq!(s, ActionState::Completed);
        assert!(s.is_terminal());
        // Terminal states reject all further transitions
        assert!(s.invalidate().is_err());
        assert!(s.reject().is_err());
    }

    #[test]
    fn c_abi_four_function_end_to_end_and_p0_preemption() {
        unsafe {
            let cp = ago_cp_create(4, 3);
            assert!(!cp.is_null());

            let budget = AgoCBudget {
                now_ms: 0.0,
                next_deadline_ms: 10.5,
                reserved_ms: 0.0,
                inflight_ms: 0.0,
                safety_margin_ms: 0.5,
            };
            assert_eq!(ago_cp_set_budget(cp, &budget), 0);

            // 1. Ingest pose revision event (bumps pose_version 1 -> 2, timestamp 100.0ms)
            let ev = AgoCEvent {
                kind: EVIDENCE_POSE_REVISION,
                critical: 1,
                region_id: 7,
                pose_version: 2,
                timestamp_ms: 100.0,
                magnitude: 2.5,
                ..Default::default()
            };
            assert_eq!(ago_ingest_events(cp, &ev, 1), 1);

            // Enqueue stale action (pose_version=1), P2 high-gain, and P0 consistency action
            let actions = [
                AgoCAction {
                    action_id: 1,
                    region_id: 7,
                    service_class: SERVICE_P2_REFINEMENT,
                    op: OP_REFINE,
                    requires_gpu: 1,
                    required_pose_version: 1,
                    expected_gain: 999.0,
                    estimated_cost_ms: 0.5,
                    ..Default::default()
                },
                AgoCAction {
                    action_id: 2,
                    region_id: 7,
                    service_class: SERVICE_P2_REFINEMENT,
                    op: OP_REFINE,
                    requires_gpu: 1,
                    required_pose_version: 2,
                    expected_gain: 100.0,
                    estimated_cost_ms: 0.5,
                    ..Default::default()
                },
                AgoCAction {
                    action_id: 3,
                    region_id: 7,
                    service_class: SERVICE_P0_CONSISTENCY,
                    op: OP_RETRACT,
                    requires_gpu: 1,
                    required_pose_version: 2,
                    expected_gain: 0.1,
                    estimated_cost_ms: 0.5,
                    ..Default::default()
                },
            ];
            assert_eq!(ago_cp_enqueue_actions(cp, actions.as_ptr(), actions.len()), 3);

            // 2. Schedule batch of max 1 -> must prune stale #1 and pick P0 #3 over P2 #2!
            let mut out = [AgoCAction::default(); 2];
            assert_eq!(ago_schedule_batch(cp, out.as_mut_ptr(), 1), 1);
            assert_eq!(out[0].action_id, 3);
            assert_eq!(out[0].evidence_ingested_at_ms, 100.0);

            // 3. Ingest completion at 104.25 ms -> event_to_map_correction_ms == 4.25 ms
            let comp = AgoCCompletion {
                action_id: 3,
                region_id: 7,
                op: OP_RETRACT,
                service_class: SERVICE_P0_CONSISTENCY,
                status: COMPLETION_OK,
                actual_cost_ms: 0.4,
                completed_at_ms: 104.25,
                ..Default::default()
            };
            assert_eq!(ago_ingest_completions(cp, &comp, 1), 1);

            // 4. Query resource summary
            let mut sum = AgoCResourceSummary::default();
            assert_eq!(ago_get_resource_summary(cp, &mut sum), 0);
            assert_eq!(sum.pose_version, 2);
            assert_eq!(sum.map_version, 2); // RETRACT is structural -> bumps map_version
            assert_eq!(sum.total_rejected_stale, 1);
            assert_eq!(sum.total_admitted, 1);
            assert_eq!(sum.total_completed, 1);
            assert!((sum.last_event_to_map_correction_ms - 4.25).abs() < 1e-6);

            ago_cp_destroy(cp);
        }
    }

    #[test]
    fn p4_maintenance_floor_and_freeze_reopen_gating() {
        let mut cp = ControlPlane::new(4, 2);
        cp.set_budget(AgoCBudget {
            now_ms: 0.0,
            next_deadline_ms: 20.5,
            reserved_ms: 0.0,
            inflight_ms: 0.0,
            safety_margin_ms: 0.5,
        });

        // Freeze region 10, then verify REFINE is blocked until REOPEN completes
        cp.enqueue_actions(&[
            AgoCAction {
                action_id: 10,
                region_id: 10,
                service_class: SERVICE_P0_CONSISTENCY,
                op: OP_FREEZE,
                requires_gpu: 1,
                expected_gain: 1.0,
                estimated_cost_ms: 0.2,
                ..Default::default()
            },
            AgoCAction {
                action_id: 11,
                region_id: 10,
                service_class: SERVICE_P2_REFINEMENT,
                op: OP_REFINE,
                requires_gpu: 1,
                expected_gain: 50.0,
                estimated_cost_ms: 0.2,
                ..Default::default()
            },
            AgoCAction {
                action_id: 99,
                region_id: 20,
                service_class: SERVICE_P4_MAINTENANCE,
                op: OP_SERIALIZE,
                requires_gpu: 1,
                expected_gain: 0.1,
                estimated_cost_ms: 0.2,
                ..Default::default()
            },
        ]);

        let mut out = [AgoCAction::default(); 1];
        assert_eq!(cp.schedule_batch(&mut out), 1);
        assert_eq!(out[0].action_id, 10);
        cp.ingest_completions(&[AgoCCompletion {
            action_id: 10,
            region_id: 10,
            op: OP_FREEZE,
            status: COMPLETION_OK,
            actual_cost_ms: 0.2,
            ..Default::default()
        }]);

        // Tick 2: region 10 is frozen so #11 is ineligible; P4 #99 runs
        assert_eq!(cp.schedule_batch(&mut out), 1);
        assert_eq!(out[0].action_id, 99);

        // Enqueue REOPEN on region 10
        cp.enqueue_actions(&[AgoCAction {
            action_id: 12,
            region_id: 10,
            service_class: SERVICE_P0_CONSISTENCY,
            op: OP_REOPEN,
            requires_gpu: 1,
            expected_gain: 1.0,
            estimated_cost_ms: 0.2,
            ..Default::default()
        }]);
        assert_eq!(cp.schedule_batch(&mut out), 1);
        assert_eq!(out[0].action_id, 12);
        cp.ingest_completions(&[AgoCCompletion {
            action_id: 12,
            region_id: 10,
            op: OP_REOPEN,
            status: COMPLETION_OK,
            actual_cost_ms: 0.2,
            ..Default::default()
        }]);

        // Now #11 is eligible and admitted!
        assert_eq!(cp.schedule_batch(&mut out), 1);
        assert_eq!(out[0].action_id, 11);
    }

    #[test]
    fn c_abi_null_pointer_safety() {
        unsafe {
            assert_eq!(ago_ingest_events(std::ptr::null_mut(), std::ptr::null(), 0), 0);
            assert_eq!(ago_schedule_batch(std::ptr::null_mut(), std::ptr::null_mut(), 0), 0);
            assert_eq!(ago_ingest_completions(std::ptr::null_mut(), std::ptr::null(), 0), 0);
            assert_eq!(ago_get_resource_summary(std::ptr::null(), std::ptr::null_mut()), -1);
        }
    }
}
