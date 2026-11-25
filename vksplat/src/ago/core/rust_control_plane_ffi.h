#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to Rust ago_control_plane::ControlPlane.
struct AgoRustControlPlane;

struct AgoRustBudget {
    double now_ms;
    double next_deadline_ms;
    double reserved_ms;
    double inflight_ms;
    double safety_margin_ms;
};

struct AgoRustEvent {
    uint8_t kind;
    uint8_t critical;
    uint8_t _pad0[2];
    uint32_t flags;
    uint64_t region_id;
    uint64_t map_version;
    uint64_t pose_version;
    uint64_t evidence_version;
    double timestamp_ms;
    float magnitude;
    uint32_t _pad1;
};

struct AgoRustAction {
    uint64_t action_id;
    uint64_t region_id;
    uint8_t service_class;
    uint8_t op;
    uint8_t requires_exclusive_region;
    uint8_t requires_gpu;
    uint32_t max_splats;
    uint64_t required_map_version;
    uint64_t required_pose_version;
    uint64_t required_evidence_version;
    float expected_gain;
    float estimated_cost_ms;
    float target_fraction;
    uint32_t _pad0;
    double enqueued_at_ms;
    double evidence_ingested_at_ms;
    double deadline_ms;
};

struct AgoRustCompletion {
    uint64_t action_id;
    uint64_t region_id;
    uint8_t op;
    uint8_t service_class;
    uint8_t status;
    uint8_t was_exclusive;
    uint32_t affected_splats;
    float realized_gain;
    float actual_cost_ms;
    double completed_at_ms;
    double event_to_map_correction_ms;
    uint64_t result_map_version;
    uint64_t result_pose_version;
};

struct AgoRustResourceSummary {
    uint64_t tick_count;
    uint64_t map_version;
    uint64_t pose_version;
    uint64_t evidence_version;
    uint32_t inflight_count;
    uint32_t queued_actions;
    uint32_t dirty_regions;
    uint32_t available_credits;
    uint64_t total_admitted;
    uint64_t total_completed;
    uint64_t total_rejected_stale;
    uint64_t total_rejected_budget;
    uint64_t total_rejected_credits;
    uint64_t total_failed;
    double available_budget_ms;
    double last_event_to_map_correction_ms;
    double max_event_to_map_correction_ms;
    double mean_event_to_map_correction_ms;
};

AgoRustControlPlane* ago_cp_create(uint32_t max_inflight, uint32_t maintenance_floor_ticks);
void ago_cp_destroy(AgoRustControlPlane* handle);
int32_t ago_cp_set_budget(AgoRustControlPlane* handle, const AgoRustBudget* budget);
uint32_t ago_cp_enqueue_actions(
    AgoRustControlPlane* handle,
    const AgoRustAction* actions_ptr,
    size_t action_count
);
uint32_t ago_ingest_events(
    AgoRustControlPlane* handle,
    const AgoRustEvent* events_ptr,
    size_t event_count
);
uint32_t ago_schedule_batch(
    AgoRustControlPlane* handle,
    AgoRustAction* out_actions_ptr,
    size_t max_actions
);
uint32_t ago_ingest_completions(
    AgoRustControlPlane* handle,
    const AgoRustCompletion* completions_ptr,
    size_t completion_count
);
int32_t ago_get_resource_summary(
    const AgoRustControlPlane* handle,
    AgoRustResourceSummary* out_summary_ptr
);

#ifdef __cplusplus
}  // extern "C"

static_assert(sizeof(AgoRustBudget) == 40, "AgoRustBudget ABI mismatch");
static_assert(sizeof(AgoRustEvent) == 56, "AgoRustEvent ABI mismatch");
static_assert(sizeof(AgoRustAction) == 88, "AgoRustAction ABI mismatch");
static_assert(sizeof(AgoRustCompletion) == 64, "AgoRustCompletion ABI mismatch");
static_assert(sizeof(AgoRustResourceSummary) == 128, "AgoRustResourceSummary ABI mismatch");
#endif
