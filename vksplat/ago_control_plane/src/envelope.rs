//! Resource envelope, 3-stage deadline/utility admission, and C-ABI POD types.

use std::collections::HashMap;

use crate::state_machine::{
    ActionState, RegionState, DIRTY_CONTRADICTION, DIRTY_EVIDENCE, DIRTY_POSE, DIRTY_STRUCTURAL,
};

// EvidenceKind discriminants matching vksplat::ago::EvidenceKind
pub const EVIDENCE_FRAME_OBSERVATION: u8 = 0;
pub const EVIDENCE_POSE_REVISION: u8 = 1;
pub const EVIDENCE_CONTRADICTION: u8 = 2;
pub const EVIDENCE_STRUCTURAL_COMPLETION: u8 = 5;

// ServiceClass discriminants matching vksplat::ago::ServiceClass
pub const SERVICE_P0_CONSISTENCY: u8 = 0;
pub const SERVICE_P1_LIVE_MAP: u8 = 1;
pub const SERVICE_P2_REFINEMENT: u8 = 2;
pub const SERVICE_P3_PREDICTIVE: u8 = 3;
pub const SERVICE_P4_MAINTENANCE: u8 = 4;

// OperationType discriminants matching vksplat::ago::OperationType
pub const OP_REFINE: u8 = 0;
pub const OP_FREEZE: u8 = 1;
pub const OP_REOPEN: u8 = 2;
pub const OP_INSERT: u8 = 3;
pub const OP_PRUNE: u8 = 7;
pub const OP_RETRACT: u8 = 10;
pub const OP_SERIALIZE: u8 = 14;
pub const OP_RESTORE: u8 = 15;
pub const OP_EVICT: u8 = 17;

// CompletionStatus discriminants matching vksplat::ago::CompletionStatus
pub const COMPLETION_OK: u8 = 0;

pub fn is_structural_op(op: u8) -> bool {
    matches!(op, 3..=12 | 15 | 17)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgoCBudget {
    pub now_ms: f64,
    pub next_deadline_ms: f64,
    pub reserved_ms: f64,
    pub inflight_ms: f64,
    pub safety_margin_ms: f64,
}

impl AgoCBudget {
    pub fn available_ms(&self) -> f64 {
        (self.next_deadline_ms - self.now_ms - self.reserved_ms - self.inflight_ms - self.safety_margin_ms)
            .max(0.0)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgoCEvent {
    pub kind: u8,
    pub critical: u8,
    pub _pad0: [u8; 2],
    pub flags: u32,
    pub region_id: u64,
    pub map_version: u64,
    pub pose_version: u64,
    pub evidence_version: u64,
    pub timestamp_ms: f64,
    pub magnitude: f32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgoCAction {
    pub action_id: u64,
    pub region_id: u64,
    pub service_class: u8,
    pub op: u8,
    pub requires_exclusive_region: u8,
    pub requires_gpu: u8,
    pub max_splats: u32,
    pub required_map_version: u64,
    pub required_pose_version: u64,
    pub required_evidence_version: u64,
    pub expected_gain: f32,
    pub estimated_cost_ms: f32,
    pub target_fraction: f32,
    pub _pad0: u32,
    pub enqueued_at_ms: f64,
    pub evidence_ingested_at_ms: f64,
    pub deadline_ms: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgoCCompletion {
    pub action_id: u64,
    pub region_id: u64,
    pub op: u8,
    pub service_class: u8,
    pub status: u8,
    pub was_exclusive: u8,
    pub affected_splats: u32,
    pub realized_gain: f32,
    pub actual_cost_ms: f32,
    pub completed_at_ms: f64,
    pub event_to_map_correction_ms: f64,
    pub result_map_version: u64,
    pub result_pose_version: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgoCResourceSummary {
    pub tick_count: u64,
    pub map_version: u64,
    pub pose_version: u64,
    pub evidence_version: u64,
    pub inflight_count: u32,
    pub queued_actions: u32,
    pub dirty_regions: u32,
    pub available_credits: u32,
    pub total_admitted: u64,
    pub total_completed: u64,
    pub total_rejected_stale: u64,
    pub total_rejected_budget: u64,
    pub total_rejected_credits: u64,
    pub total_failed: u64,
    pub available_budget_ms: f64,
    pub last_event_to_map_correction_ms: f64,
    pub max_event_to_map_correction_ms: f64,
    pub mean_event_to_map_correction_ms: f64,
}

#[derive(Debug, Clone)]
struct TrackedAction {
    action: AgoCAction,
    state: ActionState,
}

pub struct ControlPlane {
    max_inflight: u32,
    maintenance_floor_ticks: u32,
    min_cost_epsilon_ms: f32,

    tick_count: u64,
    map_version: u64,
    pose_version: u64,
    evidence_version: u64,
    ticks_since_maintenance: u64,
    next_ticket: u64,
    next_action_id: u64,

    budget: AgoCBudget,
    inflight_cost_ms: f64,

    regions: HashMap<u64, RegionState>,
    pending: Vec<TrackedAction>,
    inflight: HashMap<u64, TrackedAction>,

    total_admitted: u64,
    total_completed: u64,
    total_rejected_stale: u64,
    total_rejected_budget: u64,
    total_rejected_credits: u64,
    total_failed: u64,

    last_e2m_ms: f64,
    max_e2m_ms: f64,
    sum_e2m_ms: f64,
    e2m_samples: u64,
}

impl ControlPlane {
    pub fn new(max_inflight: u32, maintenance_floor_ticks: u32) -> Self {
        Self {
            max_inflight: max_inflight.max(1),
            maintenance_floor_ticks,
            min_cost_epsilon_ms: 1e-4,
            tick_count: 0,
            map_version: 1,
            pose_version: 1,
            evidence_version: 0,
            ticks_since_maintenance: 0,
            next_ticket: 1,
            next_action_id: 1,
            budget: AgoCBudget {
                now_ms: 0.0,
                next_deadline_ms: 16.666,
                reserved_ms: 0.0,
                inflight_ms: 0.0,
                safety_margin_ms: 0.5,
            },
            inflight_cost_ms: 0.0,
            regions: HashMap::with_capacity(256),
            pending: Vec::with_capacity(512),
            inflight: HashMap::with_capacity(max_inflight.max(4) as usize),
            total_admitted: 0,
            total_completed: 0,
            total_rejected_stale: 0,
            total_rejected_budget: 0,
            total_rejected_credits: 0,
            total_failed: 0,
            last_e2m_ms: 0.0,
            max_e2m_ms: 0.0,
            sum_e2m_ms: 0.0,
            e2m_samples: 0,
        }
    }

    pub fn set_budget(&mut self, budget: AgoCBudget) {
        self.budget = budget;
    }

    pub fn ingest_events(&mut self, events: &[AgoCEvent]) -> u32 {
        let mut processed = 0u32;
        for ev in events {
            if ev.evidence_version > self.evidence_version {
                self.evidence_version = ev.evidence_version;
            } else {
                self.evidence_version += 1;
            }

            let mut dirty_bits = DIRTY_EVIDENCE;
            match ev.kind {
                EVIDENCE_POSE_REVISION => {
                    self.pose_version = if ev.pose_version > self.pose_version {
                        ev.pose_version
                    } else {
                        self.pose_version + 1
                    };
                    dirty_bits |= DIRTY_POSE;
                }
                EVIDENCE_CONTRADICTION => {
                    dirty_bits |= DIRTY_CONTRADICTION;
                }
                EVIDENCE_STRUCTURAL_COMPLETION => {
                    self.map_version = if ev.map_version > self.map_version {
                        ev.map_version
                    } else {
                        self.map_version + 1
                    };
                    dirty_bits |= DIRTY_STRUCTURAL;
                }
                _ => {}
            }

            if ev.region_id != 0 {
                let st = self.regions.entry(ev.region_id).or_default();
                st.dirty_flags |= dirty_bits;
                st.pending_gain += ev.magnitude.max(0.0);
                st.map_version = self.map_version;
                st.pose_version = self.pose_version;
                st.evidence_version = self.evidence_version;
                if ev.timestamp_ms > 0.0 {
                    st.last_evidence_ingested_ms = ev.timestamp_ms;
                }
            }
            processed += 1;
        }
        processed
    }

    pub fn enqueue_actions(&mut self, actions: &[AgoCAction]) -> u32 {
        let mut accepted = 0u32;
        for &raw in actions {
            if !raw.estimated_cost_ms.is_finite()
                || raw.estimated_cost_ms <= 0.0
                || !raw.expected_gain.is_finite()
            {
                continue;
            }
            let mut act = raw;
            if act.action_id == 0 {
                act.action_id = self.next_action_id;
                self.next_action_id += 1;
            } else if act.action_id >= self.next_action_id {
                self.next_action_id = act.action_id + 1;
            }
            if act.evidence_ingested_at_ms <= 0.0 && act.region_id != 0 {
                if let Some(st) = self.regions.get(&act.region_id) {
                    if st.last_evidence_ingested_ms > 0.0 {
                        act.evidence_ingested_at_ms = st.last_evidence_ingested_ms;
                    }
                }
            }
            if act.evidence_ingested_at_ms <= 0.0 {
                act.evidence_ingested_at_ms = act.enqueued_at_ms;
            }
            self.pending.push(TrackedAction {
                action: act,
                state: ActionState::Candidate,
            });
            accepted += 1;
        }
        accepted
    }

    fn is_stale(&self, act: &AgoCAction) -> bool {
        let stale_map = act.required_map_version != 0 && act.required_map_version != self.map_version;
        let stale_pose = act.required_pose_version != 0 && act.required_pose_version != self.pose_version;
        stale_map || stale_pose
    }

    fn prune_stale_pending(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.is_stale(&self.pending[i].action) {
                let item = self.pending.swap_remove(i);
                let _ = item.state.invalidate();
                self.total_rejected_stale += 1;
            } else {
                i += 1;
            }
        }
    }

    fn is_region_eligible(&self, act: &AgoCAction) -> bool {
        if act.required_evidence_version != 0 && act.required_evidence_version > self.evidence_version {
            return false;
        }
        if act.region_id == 0 {
            return true;
        }
        if let Some(st) = self.regions.get(&act.region_id) {
            if st.exclusive_locked {
                return false;
            }
            if act.requires_exclusive_region != 0 && st.inflight_ops > 0 {
                return false;
            }
            if st.retracted && act.op != OP_RESTORE && act.op != OP_EVICT {
                return false;
            }
            if st.frozen
                && act.op != OP_REOPEN
                && act.op != OP_EVICT
                && act.op != OP_SERIALIZE
                && act.op != OP_RETRACT
            {
                return false;
            }
        }
        true
    }

    fn available_credits(&self) -> u32 {
        let inflight = self.inflight.len() as u32;
        self.max_inflight.saturating_sub(inflight)
    }

    pub fn schedule_batch(&mut self, out: &mut [AgoCAction]) -> u32 {
        self.tick_count += 1;
        self.ticks_since_maintenance += 1;
        self.prune_stale_pending();

        let max_out = out.len();
        let mut emitted = 0usize;

        while emitted < max_out && !self.pending.is_empty() {
            let credits = self.available_credits();
            let mut eff_budget = self.budget;
            eff_budget.inflight_ms = self.inflight_cost_ms;
            let avail_ms = eff_budget.available_ms();

            let mut best_idx: [Option<usize>; 5] = [None; 5];
            let mut best_util: [f32; 5] = [f32::NEG_INFINITY; 5];
            let mut saw_no_credits = false;
            let mut budget_rejections = 0u64;

            for idx in 0..self.pending.len() {
                let act = self.pending[idx].action;
                if act.requires_gpu != 0 && credits == 0 {
                    saw_no_credits = true;
                    continue;
                }
                if !self.is_region_eligible(&act) {
                    continue;
                }
                // Transition Candidate -> Ready via typestate
                if let Ok(ready) = self.pending[idx].state.mark_ready() {
                    self.pending[idx].state = ready;
                } else {
                    continue;
                }

                // Stage 2: Deadline admission
                if (act.estimated_cost_ms as f64) > avail_ms {
                    budget_rejections += 1;
                    continue;
                }

                // Stage 3: Marginal utility
                let cls = (act.service_class as usize).min(4);
                let denom = act.estimated_cost_ms.max(self.min_cost_epsilon_ms);
                let util = act.expected_gain.max(0.0) / denom;
                let better = match best_idx[cls] {
                    None => true,
                    Some(prev) => {
                        util > best_util[cls]
                            || (util == best_util[cls]
                                && act.action_id < self.pending[prev].action.action_id)
                    }
                };
                if better {
                    best_idx[cls] = Some(idx);
                    best_util[cls] = util;
                }
            }

            let chosen_idx = if let Some(i) = best_idx[SERVICE_P0_CONSISTENCY as usize] {
                Some(i)
            } else if self.maintenance_floor_ticks > 0
                && self.ticks_since_maintenance >= self.maintenance_floor_ticks as u64
                && best_idx[SERVICE_P4_MAINTENANCE as usize].is_some()
            {
                best_idx[SERVICE_P4_MAINTENANCE as usize]
            } else {
                (1..5).find_map(|c| best_idx[c])
            };

            let Some(sel) = chosen_idx else {
                if budget_rejections > 0 {
                    self.total_rejected_budget += budget_rejections;
                }
                if saw_no_credits {
                    self.total_rejected_credits += 1;
                }
                break;
            };

            let mut item = self.pending.swap_remove(sel);
            let ticket = self.next_ticket;
            self.next_ticket += 1;
            if let Ok(submitted) = item.state.submit(ticket) {
                item.state = submitted;
            } else {
                let _ = item.state.reject();
                self.total_failed += 1;
                continue;
            }

            let act = item.action;
            if act.region_id != 0 {
                let st = self.regions.entry(act.region_id).or_default();
                st.inflight_ops += 1;
                if act.requires_exclusive_region != 0 {
                    st.exclusive_locked = true;
                }
            }
            if act.service_class == SERVICE_P4_MAINTENANCE {
                self.ticks_since_maintenance = 0;
            }
            self.inflight_cost_ms += act.estimated_cost_ms as f64;
            self.total_admitted += 1;
            self.inflight.insert(act.action_id, item);

            out[emitted] = act;
            emitted += 1;
        }

        emitted as u32
    }

    pub fn ingest_completions(&mut self, completions: &[AgoCCompletion]) -> u32 {
        let mut processed = 0u32;
        for comp in completions {
            let ok = comp.status == COMPLETION_OK;
            let tracked = self.inflight.remove(&comp.action_id);
            let mut e2m_ms = comp.event_to_map_correction_ms;

            if let Some(item) = tracked {
                self.inflight_cost_ms = (self.inflight_cost_ms - item.action.estimated_cost_ms as f64).max(0.0);
                if ok {
                    let _ = item.state.complete();
                } else {
                    let _ = item.state.reject();
                }
                if e2m_ms <= 0.0 && comp.completed_at_ms > 0.0 && item.action.evidence_ingested_at_ms > 0.0 {
                    e2m_ms = (comp.completed_at_ms - item.action.evidence_ingested_at_ms).max(0.0);
                }
            }

            if comp.region_id != 0 {
                let st = self.regions.entry(comp.region_id).or_default();
                st.inflight_ops = st.inflight_ops.saturating_sub(1);
                if comp.was_exclusive != 0 {
                    st.exclusive_locked = false;
                }
                if ok {
                    st.completed_ops += 1;
                    match comp.op {
                        OP_FREEZE => st.frozen = true,
                        OP_REOPEN => st.frozen = false,
                        OP_RETRACT | OP_PRUNE => st.retracted = true,
                        OP_RESTORE => st.retracted = false,
                        _ => {}
                    }
                    st.dirty_flags &= !(DIRTY_EVIDENCE | DIRTY_CONTRADICTION);
                }
            }

            if ok {
                self.total_completed += 1;
                if e2m_ms <= 0.0 {
                    e2m_ms = (comp.actual_cost_ms as f64).max(0.0);
                }
                self.last_e2m_ms = e2m_ms;
                if e2m_ms > self.max_e2m_ms {
                    self.max_e2m_ms = e2m_ms;
                }
                self.sum_e2m_ms += e2m_ms;
                self.e2m_samples += 1;

                if comp.result_map_version > self.map_version {
                    self.map_version = comp.result_map_version;
                } else if is_structural_op(comp.op) {
                    self.map_version += 1;
                }
            } else {
                self.total_failed += 1;
            }
            processed += 1;
        }
        processed
    }

    pub fn resource_summary(&self) -> AgoCResourceSummary {
        let mut eff_budget = self.budget;
        eff_budget.inflight_ms = self.inflight_cost_ms;
        let dirty_regions = self
            .regions
            .values()
            .filter(|r| r.dirty_flags != 0)
            .count() as u32;
        let mean_e2m = if self.e2m_samples > 0 {
            self.sum_e2m_ms / (self.e2m_samples as f64)
        } else {
            0.0
        };
        AgoCResourceSummary {
            tick_count: self.tick_count,
            map_version: self.map_version,
            pose_version: self.pose_version,
            evidence_version: self.evidence_version,
            inflight_count: self.inflight.len() as u32,
            queued_actions: self.pending.len() as u32,
            dirty_regions,
            available_credits: self.available_credits(),
            total_admitted: self.total_admitted,
            total_completed: self.total_completed,
            total_rejected_stale: self.total_rejected_stale,
            total_rejected_budget: self.total_rejected_budget,
            total_rejected_credits: self.total_rejected_credits,
            total_failed: self.total_failed,
            available_budget_ms: eff_budget.available_ms(),
            last_event_to_map_correction_ms: self.last_e2m_ms,
            max_event_to_map_correction_ms: self.max_e2m_ms,
            mean_event_to_map_correction_ms: mean_e2m,
        }
    }
}
