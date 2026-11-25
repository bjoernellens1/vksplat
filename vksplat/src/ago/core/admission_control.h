#pragma once

#include "action_protocol.h"
#include "../spatial/dirty_tracking.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace vksplat::ago {

struct DeadlineBudget {
    double now_ms = 0.0;
    double next_deadline_ms = 16.666;
    double reserved_ms = 0.0;       // C_reserved
    double inflight_ms = 0.0;       // C_inflight
    double safety_margin_ms = 0.5;  // m_t

    double available_ms() const noexcept {
        return std::max(0.0, next_deadline_ms - now_ms - reserved_ms - inflight_ms - safety_margin_ms);
    }
};

struct AdmissionConfig {
    uint32_t max_inflight_actions = 4;
    uint32_t maintenance_floor_interval_ticks = 8;
    float min_cost_epsilon_ms = 1e-4f;
};

struct AdmissionContext {
    uint64_t current_map_version = 1;
    uint64_t current_pose_version = 1;
    uint64_t current_evidence_version = 0;
    uint32_t available_credits = 0;
    uint64_t current_tick = 0;
    uint64_t ticks_since_last_maintenance = 0;
    DeadlineBudget budget{};
};

enum class EligibilityResult : uint8_t {
    ELIGIBLE = 0,
    STALE_VERSION = 1,
    NOT_READY = 2,
    NO_CREDITS = 3,
    REGION_LOCKED = 4,
    REGION_FROZEN = 5,
    REGION_RETRACTED = 6,
    INVALID = 7,
};

struct AdmissionDecision {
    bool admitted = false;
    size_t selected_index = 0;
    float marginal_utility = 0.0f;
    bool rejected_by_budget = false;
    bool rejected_by_credits = false;
    size_t budget_rejections = 0;
};

// 3-stage non-blocking admission controller:
//   Stage 1: Eligibility (ready && valid && resources)
//   Stage 2: Deadline Admission (B_t = max(0, d_next - t - C_reserved - C_inflight - m_t))
//   Stage 3: Marginal Utility (argmax(expected_gain / estimated_cost_ms) respecting P0..P4 + P4 floor)
class AdmissionController {
public:
    explicit AdmissionController(const AdmissionConfig& config = {}) noexcept
        : config_(config) {}

    const AdmissionConfig& config() const noexcept { return config_; }
    void set_config(const AdmissionConfig& cfg) noexcept { config_ = cfg; }

    EligibilityResult check_eligibility(
        const AGOAction& action,
        const AdmissionContext& ctx,
        const DirtyTracker& tracker
    ) const noexcept;

    bool check_deadline_admission(
        const AGOAction& action,
        const AdmissionContext& ctx
    ) const noexcept;

    float compute_marginal_utility(const AGOAction& action) const noexcept;

    AdmissionDecision select_next(
        const AGOAction* candidates,
        size_t candidate_count,
        const AdmissionContext& ctx,
        const DirtyTracker& tracker
    ) const noexcept;

private:
    AdmissionConfig config_{};
};

}  // namespace vksplat::ago
