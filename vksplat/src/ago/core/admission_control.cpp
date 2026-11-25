#include "admission_control.h"

#include <array>
#include <cmath>
#include <limits>

namespace vksplat::ago {

EligibilityResult AdmissionController::check_eligibility(
    const AGOAction& action,
    const AdmissionContext& ctx,
    const DirtyTracker& tracker
) const noexcept {
    if (!std::isfinite(action.estimated_cost_ms) || action.estimated_cost_ms <= 0.0f ||
        !std::isfinite(action.expected_gain)) {
        return EligibilityResult::INVALID;
    }

    // Version compatibility check
    if (action.required_map_version != 0u &&
        action.required_map_version != ctx.current_map_version) {
        return EligibilityResult::STALE_VERSION;
    }
    if (action.required_pose_version != 0u &&
        action.required_pose_version != ctx.current_pose_version) {
        return EligibilityResult::STALE_VERSION;
    }
    if (action.required_evidence_version != 0u &&
        action.required_evidence_version > ctx.current_evidence_version) {
        return EligibilityResult::NOT_READY;
    }

    // Resource check
    if (action.requires_gpu && ctx.available_credits == 0u) {
        return EligibilityResult::NO_CREDITS;
    }

    // Region state check (sparse lookup, no full-map scan)
    if (action.region_id != 0u) {
        const RegionWorkState* st = tracker.find(action.region_id);
        if (st) {
            if (st->exclusive_locked) {
                return EligibilityResult::REGION_LOCKED;
            }
            if (action.requires_exclusive_region && st->inflight_ops > 0u) {
                return EligibilityResult::REGION_LOCKED;
            }
            if (st->retracted &&
                action.op != OperationType::RESTORE &&
                action.op != OperationType::EVICT) {
                return EligibilityResult::REGION_RETRACTED;
            }
            if (st->frozen &&
                action.op != OperationType::REOPEN &&
                action.op != OperationType::EVICT &&
                action.op != OperationType::SERIALIZE &&
                action.op != OperationType::RETRACT) {
                return EligibilityResult::REGION_FROZEN;
            }
        }
    }

    return EligibilityResult::ELIGIBLE;
}

bool AdmissionController::check_deadline_admission(
    const AGOAction& action,
    const AdmissionContext& ctx
) const noexcept {
    const double available_ms = ctx.budget.available_ms();
    return static_cast<double>(action.estimated_cost_ms) <= available_ms;
}

float AdmissionController::compute_marginal_utility(const AGOAction& action) const noexcept {
    const float denom = std::max(action.estimated_cost_ms, config_.min_cost_epsilon_ms);
    const float gain = std::max(0.0f, action.expected_gain);
    return gain / denom;
}

AdmissionDecision AdmissionController::select_next(
    const AGOAction* candidates,
    size_t candidate_count,
    const AdmissionContext& ctx,
    const DirtyTracker& tracker
) const noexcept {
    AdmissionDecision decision{};
    if (!candidates || candidate_count == 0u) {
        return decision;
    }

    static constexpr size_t kNumClasses = 5u;
    std::array<bool, kNumClasses> has_best = {false, false, false, false, false};
    std::array<size_t, kNumClasses> best_idx = {0, 0, 0, 0, 0};
    std::array<float, kNumClasses> best_util = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };

    bool saw_no_credits = false;

    for (size_t i = 0; i < candidate_count; ++i) {
        const AGOAction& act = candidates[i];

        // Stage 1: Eligibility
        const EligibilityResult elig = check_eligibility(act, ctx, tracker);
        if (elig == EligibilityResult::NO_CREDITS) {
            saw_no_credits = true;
            continue;
        }
        if (elig != EligibilityResult::ELIGIBLE) {
            continue;
        }

        // Stage 2: Deadline Admission
        if (!check_deadline_admission(act, ctx)) {
            ++decision.budget_rejections;
            decision.rejected_by_budget = true;
            continue;
        }

        // Stage 3: Marginal Utility per service class
        const size_t cls = std::min<size_t>(static_cast<size_t>(act.service_class), kNumClasses - 1u);
        const float util = compute_marginal_utility(act);

        if (!has_best[cls] ||
            util > best_util[cls] ||
            (util == best_util[cls] && act.action_id < candidates[best_idx[cls]].action_id)) {
            has_best[cls] = true;
            best_idx[cls] = i;
            best_util[cls] = util;
        }
    }

    // Priority order: P0_CONSISTENCY always first.
    if (has_best[static_cast<size_t>(ServiceClass::P0_CONSISTENCY)]) {
        const size_t c = static_cast<size_t>(ServiceClass::P0_CONSISTENCY);
        decision.admitted = true;
        decision.selected_index = best_idx[c];
        decision.marginal_utility = best_util[c];
        return decision;
    }

    // Anti-starvation maintenance floor for P4_MAINTENANCE.
    const size_t p4 = static_cast<size_t>(ServiceClass::P4_MAINTENANCE);
    if (config_.maintenance_floor_interval_ticks > 0u &&
        ctx.ticks_since_last_maintenance >= config_.maintenance_floor_interval_ticks &&
        has_best[p4]) {
        decision.admitted = true;
        decision.selected_index = best_idx[p4];
        decision.marginal_utility = best_util[p4];
        return decision;
    }

    // Standard priority cascade: P1 -> P2 -> P3 -> P4
    for (size_t c = static_cast<size_t>(ServiceClass::P1_LIVE_MAP); c < kNumClasses; ++c) {
        if (has_best[c]) {
            decision.admitted = true;
            decision.selected_index = best_idx[c];
            decision.marginal_utility = best_util[c];
            return decision;
        }
    }

    if (saw_no_credits) {
        decision.rejected_by_credits = true;
    }
    return decision;
}

}  // namespace vksplat::ago
