#pragma once

#include <cstdint>
#include <cstddef>

namespace vksplat::ago {

using RegionId = uint64_t;
using ActionId = uint64_t;

enum class ServiceClass : uint8_t {
    P0_CONSISTENCY = 0,
    P1_LIVE_MAP = 1,
    P2_REFINEMENT = 2,
    P3_PREDICTIVE = 3,
    P4_MAINTENANCE = 4,
};

enum class ActiveSetState : uint8_t {
    EMPTY = 0,
    PARTIAL = 1,
    FULL = 2,
    INVALID = 3,
};

enum class OperationType : uint8_t {
    REFINE = 0,
    FREEZE = 1,
    REOPEN = 2,
    INSERT = 3,
    CLONE = 4,
    SPLIT = 5,
    MERGE = 6,
    PRUNE = 7,
    RELOCATE = 8,
    REANCHOR = 9,
    RETRACT = 10,
    TRANSFORM = 11,
    PARTITION_MOVE = 12,
    LOD_CHANGE = 13,
    SERIALIZE = 14,
    RESTORE = 15,
    PREFETCH = 16,
    EVICT = 17,
};

enum class EvidenceKind : uint8_t {
    FRAME_OBSERVATION = 0,
    POSE_REVISION = 1,
    CONTRADICTION = 2,
    COVERAGE_UPDATE = 3,
    TELEMETRY_SUMMARY = 4,
    STRUCTURAL_COMPLETION = 5,
    DEADLINE_BUDGET_UPDATE = 6,
    SHUTDOWN_SIGNAL = 7,
};

enum class CompletionStatus : uint8_t {
    OK = 0,
    REJECTED_STALE = 1,
    REJECTED_NO_CREDITS = 2,
    REJECTED_UNSUPPORTED = 3,
    REJECTED_INVALID = 4,
    FAILED_EXECUTION = 5,
    CANCELLED = 6,
};

enum class SubmitResult : uint8_t {
    OK = 0,
    NO_CREDITS = 1,
    INVALID_ACTION = 2,
    UNSUPPORTED_OP = 3,
    DEVICE_ERROR = 4,
};

// Region dirty-flag bitmask
static constexpr uint32_t DIRTY_NONE          = 0u;
static constexpr uint32_t DIRTY_EVIDENCE      = 1u << 0;
static constexpr uint32_t DIRTY_POSE          = 1u << 1;
static constexpr uint32_t DIRTY_CONTRADICTION = 1u << 2;
static constexpr uint32_t DIRTY_STRUCTURAL    = 1u << 3;
static constexpr uint32_t DIRTY_MAINTENANCE   = 1u << 4;

struct EvidenceMessage {
    EvidenceKind kind = EvidenceKind::FRAME_OBSERVATION;
    RegionId region_id = 0;
    uint64_t map_version = 0;
    uint64_t pose_version = 0;
    uint64_t evidence_version = 0;
    double timestamp_ms = 0.0;
    float magnitude = 0.0f;
    uint32_t flags = 0;
    bool critical = false;
};

struct AGOAction {
    ActionId action_id = 0;
    RegionId region_id = 0;
    ServiceClass service_class = ServiceClass::P2_REFINEMENT;
    OperationType op = OperationType::REFINE;
    uint64_t required_map_version = 0;
    uint64_t required_pose_version = 0;
    uint64_t required_evidence_version = 0;
    float expected_gain = 0.0f;
    float estimated_cost_ms = 0.1f;
    uint32_t max_splats = 0;
    float target_fraction = 0.0f;
    bool requires_exclusive_region = false;
    bool requires_gpu = true;
    double enqueued_at_ms = 0.0;
    double deadline_ms = 0.0;
};

struct CompletionRecord {
    ActionId action_id = 0;
    RegionId region_id = 0;
    OperationType op = OperationType::REFINE;
    ServiceClass service_class = ServiceClass::P2_REFINEMENT;
    CompletionStatus status = CompletionStatus::OK;
    ActiveSetState active_set_state = ActiveSetState::INVALID;
    uint32_t affected_splats = 0;
    float realized_gain = 0.0f;
    float actual_cost_ms = 0.0f;
    uint64_t result_map_version = 0;
    uint64_t result_pose_version = 0;
    uint64_t completed_at_tick = 0;
    bool was_exclusive = false;
};

struct RegionWorkState {
    RegionId region_id = 0;
    uint64_t map_version = 0;
    uint64_t pose_version = 0;
    uint64_t evidence_version = 0;
    uint32_t dirty_flags = DIRTY_NONE;
    float pending_gain = 0.0f;
    float estimated_cost_ms = 0.1f;
    uint32_t inflight_ops = 0;
    uint32_t completed_ops = 0;
    bool frozen = false;
    bool retracted = false;
    bool exclusive_locked = false;
    bool in_ready_queue = false;
    uint64_t last_updated_tick = 0;
};

inline bool is_structural_op(OperationType op) noexcept {
    switch (op) {
        case OperationType::INSERT:
        case OperationType::CLONE:
        case OperationType::SPLIT:
        case OperationType::MERGE:
        case OperationType::PRUNE:
        case OperationType::RELOCATE:
        case OperationType::REANCHOR:
        case OperationType::RETRACT:
        case OperationType::TRANSFORM:
        case OperationType::PARTITION_MOVE:
        case OperationType::RESTORE:
        case OperationType::EVICT:
            return true;
        default:
            return false;
    }
}

inline const char* to_string(ServiceClass sc) noexcept {
    switch (sc) {
        case ServiceClass::P0_CONSISTENCY: return "P0_CONSISTENCY";
        case ServiceClass::P1_LIVE_MAP:    return "P1_LIVE_MAP";
        case ServiceClass::P2_REFINEMENT:  return "P2_REFINEMENT";
        case ServiceClass::P3_PREDICTIVE:  return "P3_PREDICTIVE";
        case ServiceClass::P4_MAINTENANCE: return "P4_MAINTENANCE";
    }
    return "UNKNOWN";
}

inline const char* to_string(ActiveSetState st) noexcept {
    switch (st) {
        case ActiveSetState::EMPTY:   return "EMPTY";
        case ActiveSetState::PARTIAL: return "PARTIAL";
        case ActiveSetState::FULL:    return "FULL";
        case ActiveSetState::INVALID: return "INVALID";
    }
    return "INVALID";
}

inline const char* to_string(OperationType op) noexcept {
    switch (op) {
        case OperationType::REFINE:         return "REFINE";
        case OperationType::FREEZE:         return "FREEZE";
        case OperationType::REOPEN:         return "REOPEN";
        case OperationType::INSERT:         return "INSERT";
        case OperationType::CLONE:          return "CLONE";
        case OperationType::SPLIT:          return "SPLIT";
        case OperationType::MERGE:          return "MERGE";
        case OperationType::PRUNE:          return "PRUNE";
        case OperationType::RELOCATE:       return "RELOCATE";
        case OperationType::REANCHOR:       return "REANCHOR";
        case OperationType::RETRACT:        return "RETRACT";
        case OperationType::TRANSFORM:      return "TRANSFORM";
        case OperationType::PARTITION_MOVE: return "PARTITION_MOVE";
        case OperationType::LOD_CHANGE:     return "LOD_CHANGE";
        case OperationType::SERIALIZE:      return "SERIALIZE";
        case OperationType::RESTORE:        return "RESTORE";
        case OperationType::PREFETCH:       return "PREFETCH";
        case OperationType::EVICT:          return "EVICT";
    }
    return "UNKNOWN";
}

inline const char* to_string(CompletionStatus st) noexcept {
    switch (st) {
        case CompletionStatus::OK:                   return "OK";
        case CompletionStatus::REJECTED_STALE:       return "REJECTED_STALE";
        case CompletionStatus::REJECTED_NO_CREDITS:  return "REJECTED_NO_CREDITS";
        case CompletionStatus::REJECTED_UNSUPPORTED: return "REJECTED_UNSUPPORTED";
        case CompletionStatus::REJECTED_INVALID:     return "REJECTED_INVALID";
        case CompletionStatus::FAILED_EXECUTION:     return "FAILED_EXECUTION";
        case CompletionStatus::CANCELLED:            return "CANCELLED";
    }
    return "UNKNOWN";
}

}  // namespace vksplat::ago
