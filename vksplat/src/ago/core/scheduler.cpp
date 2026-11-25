#include "scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace vksplat::ago {

AgoScheduler::AgoScheduler(
    const SchedulerConfig& config,
    ActionExecutorInterface* executor
)
    : config_(config),
      executor_(executor),
      dirty_tracker_(config.max_regions),
      admission_(config.admission),
      trace_(config.trace_capacity),
      critical_evidence_queue_(config.critical_queue_capacity),
      telemetry_evidence_queue_(config.telemetry_queue_capacity),
      pending_actions_(std::max<size_t>(config.max_pending_actions, 16u)),
      completion_scratch_(std::max<size_t>(config.max_completions_per_tick, 1u)) {
    summary_.map_version = map_version_;
    summary_.pose_version = pose_version_;
    summary_.evidence_version = evidence_version_;
}

bool AgoScheduler::push_critical_evidence(const EvidenceMessage& msg) noexcept {
    const uint64_t ver = std::max({msg.evidence_version, msg.pose_version, msg.map_version});
    return critical_evidence_queue_.try_push(msg, ver);
}

bool AgoScheduler::push_telemetry_evidence(const EvidenceMessage& msg) noexcept {
    return telemetry_evidence_queue_.push(msg);
}

void AgoScheduler::publish_deadline_budget(const DeadlineBudget& budget) noexcept {
    budget_mailbox_.publish(budget);
}

bool AgoScheduler::enqueue_action(const AGOAction& action) noexcept {
    if (pending_count_ >= pending_actions_.size()) {
        return false;
    }
    if (!std::isfinite(action.estimated_cost_ms) || action.estimated_cost_ms <= 0.0f ||
        !std::isfinite(action.expected_gain)) {
        return false;
    }
    AGOAction copy = action;
    if (copy.action_id == 0u) {
        copy.action_id = next_action_id_++;
    } else if (copy.action_id >= next_action_id_) {
        next_action_id_ = copy.action_id + 1u;
    }
    pending_actions_[pending_count_++] = copy;
    return true;
}

void AgoScheduler::handle_evidence(const EvidenceMessage& msg, bool is_critical) noexcept {
    if (msg.evidence_version > evidence_version_) {
        evidence_version_ = msg.evidence_version;
    } else {
        ++evidence_version_;
    }

    uint32_t dirty_bits = DIRTY_EVIDENCE;
    switch (msg.kind) {
        case EvidenceKind::POSE_REVISION:
            pose_version_ = (msg.pose_version > pose_version_) ? msg.pose_version : (pose_version_ + 1u);
            dirty_bits |= DIRTY_POSE;
            break;
        case EvidenceKind::CONTRADICTION:
            dirty_bits |= DIRTY_CONTRADICTION;
            break;
        case EvidenceKind::STRUCTURAL_COMPLETION:
            map_version_ = (msg.map_version > map_version_) ? msg.map_version : (map_version_ + 1u);
            dirty_bits |= DIRTY_STRUCTURAL;
            break;
        default:
            break;
    }

    if (msg.region_id != 0u) {
        dirty_tracker_.mark_dirty(
            msg.region_id,
            dirty_bits,
            msg.magnitude,
            map_version_,
            pose_version_,
            evidence_version_,
            tick_count_
        );
    }

    EventTraceEntry ev{};
    ev.tick = tick_count_;
    ev.event_type = is_critical ? TraceEventType::EVIDENCE_CRITICAL : TraceEventType::EVIDENCE_TELEMETRY;
    ev.region_id = msg.region_id;
    ev.value = msg.magnitude;
    trace_.record(ev);
}

void AgoScheduler::handle_completion(const CompletionRecord& rec) noexcept {
    const bool ok = (rec.status == CompletionStatus::OK);
    if (rec.region_id != 0u) {
        dirty_tracker_.complete_inflight(rec.region_id, rec.was_exclusive, ok, tick_count_);
        if (ok) {
            if (rec.op == OperationType::FREEZE) {
                dirty_tracker_.set_frozen(rec.region_id, true, tick_count_);
            } else if (rec.op == OperationType::REOPEN) {
                dirty_tracker_.set_frozen(rec.region_id, false, tick_count_);
            } else if (rec.op == OperationType::RETRACT) {
                dirty_tracker_.set_retracted(rec.region_id, true, tick_count_);
            } else if (rec.op == OperationType::RESTORE) {
                dirty_tracker_.set_retracted(rec.region_id, false, tick_count_);
            }
            dirty_tracker_.clear_dirty(rec.region_id, DIRTY_EVIDENCE | DIRTY_CONTRADICTION);
        }
    }

    if (ok) {
        ++summary_.total_completed;
        if (rec.result_map_version > map_version_) {
            map_version_ = rec.result_map_version;
        } else if (is_structural_op(rec.op)) {
            ++map_version_;
        }
    } else if (rec.status == CompletionStatus::REJECTED_UNSUPPORTED) {
        ++summary_.total_rejected_unsupported;
    } else if (rec.status == CompletionStatus::FAILED_EXECUTION) {
        ++summary_.total_failed;
    }

    EventTraceEntry ev{};
    ev.tick = tick_count_;
    ev.event_type = TraceEventType::COMPLETION_IN;
    ev.action_id = rec.action_id;
    ev.region_id = rec.region_id;
    ev.service_class = rec.service_class;
    ev.op = rec.op;
    ev.status = rec.status;
    ev.value = rec.actual_cost_ms;
    trace_.record(ev);
}

uint32_t AgoScheduler::prune_stale_actions() noexcept {
    uint32_t pruned = 0u;
    size_t i = 0u;
    while (i < pending_count_) {
        const AGOAction& act = pending_actions_[i];
        const bool stale_map = (act.required_map_version != 0u && act.required_map_version != map_version_);
        const bool stale_pose = (act.required_pose_version != 0u && act.required_pose_version != pose_version_);
        if (stale_map || stale_pose) {
            EventTraceEntry ev{};
            ev.tick = tick_count_;
            ev.event_type = TraceEventType::ACTION_REJECTED_STALE;
            ev.action_id = act.action_id;
            ev.region_id = act.region_id;
            ev.service_class = act.service_class;
            ev.op = act.op;
            ev.status = CompletionStatus::REJECTED_STALE;
            trace_.record(ev);

            // O(1) swap-and-pop removal (zero allocation)
            pending_actions_[i] = pending_actions_[pending_count_ - 1u];
            --pending_count_;
            ++pruned;
            ++summary_.total_rejected_stale;
        } else {
            ++i;
        }
    }
    return pruned;
}

TickResult AgoScheduler::tick() noexcept {
    const auto t0 = std::chrono::steady_clock::now();
    ++tick_count_;
    ++ticks_since_last_maintenance_;

    TickResult result{};
    result.tick_index = tick_count_;

    // 1. Pull latest deadline budget snapshot (wait-free).
    DeadlineBudget incoming_budget{};
    if (budget_mailbox_.try_consume(incoming_budget)) {
        current_budget_ = incoming_budget;
    }

    // 2. Drain bounded completed actions from non-blocking executor.
    if (executor_) {
        const size_t n_done = executor_->poll_completed(
            completion_scratch_.data(),
            completion_scratch_.size()
        );
        for (size_t i = 0; i < n_done; ++i) {
            handle_completion(completion_scratch_[i]);
        }
        result.completions_processed = static_cast<uint32_t>(n_done);
    }

    // 3. Drain bounded critical evidence (pose revisions, contradictions, structural completions).
    const size_t n_crit = critical_evidence_queue_.drain_up_to(
        config_.max_critical_evidence_per_tick,
        [this](const EvidenceMessage& msg, uint64_t) noexcept {
            handle_evidence(msg, true);
        }
    );
    result.critical_evidence_processed = static_cast<uint32_t>(n_crit);

    // 4. Drain bounded telemetry evidence (non-critical, drop-oldest on producer overflow).
    const size_t n_tel = telemetry_evidence_queue_.drain_up_to(
        config_.max_telemetry_evidence_per_tick,
        [this](const EvidenceMessage& msg) noexcept {
            handle_evidence(msg, false);
        }
    );
    result.telemetry_evidence_processed = static_cast<uint32_t>(n_tel);

    // 5. Prune stale/incompatible pending actions after version updates.
    result.stale_actions_pruned = prune_stale_actions();

    // 6. Run bounded 3-stage admission loop (Stage 1 Eligibility -> Stage 2 Deadline -> Stage 3 Utility).
    double tick_committed_ms = 0.0;
    for (size_t adm = 0; adm < config_.max_admissions_per_tick && pending_count_ > 0u; ++adm) {
        AdmissionContext ctx{};
        ctx.current_map_version = map_version_;
        ctx.current_pose_version = pose_version_;
        ctx.current_evidence_version = evidence_version_;
        ctx.available_credits = executor_ ? executor_->available_credits() : config_.admission.max_inflight_actions;
        ctx.current_tick = tick_count_;
        ctx.ticks_since_last_maintenance = ticks_since_last_maintenance_;
        ctx.budget = current_budget_;
        ctx.budget.inflight_ms = (executor_ ? executor_->inflight_cost_ms() : 0.0) + tick_committed_ms;

        const AdmissionDecision dec = admission_.select_next(
            pending_actions_.data(),
            pending_count_,
            ctx,
            dirty_tracker_
        );

        if (!dec.admitted) {
            if (dec.rejected_by_budget) {
                result.budget_rejections += static_cast<uint32_t>(dec.budget_rejections);
                summary_.total_rejected_budget += dec.budget_rejections;
                EventTraceEntry ev{};
                ev.tick = tick_count_;
                ev.event_type = TraceEventType::ACTION_REJECTED_BUDGET;
                ev.value = static_cast<float>(ctx.budget.available_ms());
                trace_.record(ev);
            }
            if (dec.rejected_by_credits) {
                ++result.credit_rejections;
                ++summary_.total_rejected_credits;
                EventTraceEntry ev{};
                ev.tick = tick_count_;
                ev.event_type = TraceEventType::ACTION_REJECTED_CREDITS;
                trace_.record(ev);
            }
            break;
        }

        const AGOAction chosen = pending_actions_[dec.selected_index];
        SubmitResult sub_res = SubmitResult::OK;
        if (executor_) {
            sub_res = executor_->try_submit(chosen);
        }

        if (sub_res == SubmitResult::OK) {
            // Remove admitted action via swap-and-pop
            pending_actions_[dec.selected_index] = pending_actions_[pending_count_ - 1u];
            --pending_count_;

            if (chosen.region_id != 0u) {
                dirty_tracker_.mark_inflight(
                    chosen.region_id,
                    chosen.requires_exclusive_region,
                    tick_count_
                );
            }
            if (!executor_) {
                tick_committed_ms += static_cast<double>(chosen.estimated_cost_ms);
            }
            if (chosen.service_class == ServiceClass::P4_MAINTENANCE) {
                ticks_since_last_maintenance_ = 0u;
            }

            ++result.actions_admitted;
            ++summary_.total_admitted;

            EventTraceEntry ev{};
            ev.tick = tick_count_;
            ev.event_type = TraceEventType::ACTION_ADMITTED;
            ev.action_id = chosen.action_id;
            ev.region_id = chosen.region_id;
            ev.service_class = chosen.service_class;
            ev.op = chosen.op;
            ev.status = CompletionStatus::OK;
            ev.value = dec.marginal_utility;
            trace_.record(ev);
        } else if (sub_res == SubmitResult::NO_CREDITS) {
            ++result.credit_rejections;
            ++summary_.total_rejected_credits;
            EventTraceEntry ev{};
            ev.tick = tick_count_;
            ev.event_type = TraceEventType::ACTION_REJECTED_CREDITS;
            ev.action_id = chosen.action_id;
            ev.region_id = chosen.region_id;
            ev.service_class = chosen.service_class;
            ev.op = chosen.op;
            ev.status = CompletionStatus::REJECTED_NO_CREDITS;
            trace_.record(ev);
            break;
        } else if (sub_res == SubmitResult::UNSUPPORTED_OP) {
            // Drop unsupported action cleanly with explicit error trace (never silently fall back)
            pending_actions_[dec.selected_index] = pending_actions_[pending_count_ - 1u];
            --pending_count_;
            ++summary_.total_rejected_unsupported;
            EventTraceEntry ev{};
            ev.tick = tick_count_;
            ev.event_type = TraceEventType::ACTION_REJECTED_UNSUPPORTED;
            ev.action_id = chosen.action_id;
            ev.region_id = chosen.region_id;
            ev.service_class = chosen.service_class;
            ev.op = chosen.op;
            ev.status = CompletionStatus::REJECTED_UNSUPPORTED;
            trace_.record(ev);
        } else {
            // Invalid action or device error: remove from pending and record failure
            pending_actions_[dec.selected_index] = pending_actions_[pending_count_ - 1u];
            --pending_count_;
            ++summary_.total_failed;
            EventTraceEntry ev{};
            ev.tick = tick_count_;
            ev.event_type = TraceEventType::ACTION_SUBMIT_FAILED;
            ev.action_id = chosen.action_id;
            ev.region_id = chosen.region_id;
            ev.service_class = chosen.service_class;
            ev.op = chosen.op;
            ev.status = CompletionStatus::FAILED_EXECUTION;
            trace_.record(ev);
        }
    }

    // 7. Update & publish non-blocking summary snapshot.
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    result.tick_duration_us = elapsed_us;

    summary_.tick_count = tick_count_;
    summary_.map_version = map_version_;
    summary_.pose_version = pose_version_;
    summary_.evidence_version = evidence_version_;
    summary_.inflight_count = executor_ ? executor_->inflight_count() : 0u;
    summary_.queued_actions = static_cast<uint32_t>(pending_count_);
    summary_.dirty_regions = static_cast<uint32_t>(dirty_tracker_.dirty_queue_size());
    summary_.total_dropped_telemetry = telemetry_evidence_queue_.dropped_count();
    summary_.total_critical_backpressure = critical_evidence_queue_.rejected_full_count();
    summary_.last_tick_us = elapsed_us;

    DeadlineBudget effective = current_budget_;
    effective.inflight_ms = executor_ ? executor_->inflight_cost_ms() : tick_committed_ms;
    summary_.available_budget_ms = effective.available_ms();

    summary_mailbox_.publish(summary_);
    return result;
}

}  // namespace vksplat::ago
