#pragma once

#include "action_protocol.h"
#include "admission_control.h"
#include "mailbox.h"
#include "../spatial/dirty_tracking.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vksplat::ago {

enum class TraceEventType : uint8_t {
    EVIDENCE_CRITICAL = 0,
    EVIDENCE_TELEMETRY = 1,
    COMPLETION_IN = 2,
    ACTION_ADMITTED = 3,
    ACTION_REJECTED_STALE = 4,
    ACTION_REJECTED_BUDGET = 5,
    ACTION_REJECTED_CREDITS = 6,
    ACTION_REJECTED_UNSUPPORTED = 7,
    ACTION_SUBMIT_FAILED = 8,
    TICK_SUMMARY = 9,
};

struct EventTraceEntry {
    uint64_t tick = 0;
    TraceEventType event_type = TraceEventType::TICK_SUMMARY;
    ActionId action_id = 0;
    RegionId region_id = 0;
    ServiceClass service_class = ServiceClass::P2_REFINEMENT;
    OperationType op = OperationType::REFINE;
    CompletionStatus status = CompletionStatus::OK;
    float value = 0.0f;
};

// Preallocated deterministic ring buffer for scheduler events (zero hot-path allocations).
class EventTrace {
public:
    explicit EventTrace(size_t capacity = 4096)
        : capacity_(capacity > 0 ? capacity : 64),
          ring_(capacity_) {}

    void record(const EventTraceEntry& entry) noexcept {
        ring_[write_pos_] = entry;
        write_pos_ = (write_pos_ + 1u) % capacity_;
        if (count_ < capacity_) {
            ++count_;
        }
        ++total_events_;
    }

    size_t size() const noexcept { return count_; }
    size_t capacity() const noexcept { return capacity_; }
    uint64_t total_events() const noexcept { return total_events_; }

    // Copies chronological trace entries into `out` (outside hot path).
    void snapshot(std::vector<EventTraceEntry>& out) const {
        out.clear();
        out.reserve(count_);
        const size_t start = (count_ == capacity_) ? write_pos_ : 0u;
        for (size_t i = 0; i < count_; ++i) {
            out.push_back(ring_[(start + i) % capacity_]);
        }
    }

    void clear() noexcept {
        write_pos_ = 0;
        count_ = 0;
        total_events_ = 0;
    }

private:
    size_t capacity_ = 64;
    std::vector<EventTraceEntry> ring_;
    size_t write_pos_ = 0;
    size_t count_ = 0;
    uint64_t total_events_ = 0;
};

struct SchedulerSummary {
    uint64_t tick_count = 0;
    uint64_t map_version = 1;
    uint64_t pose_version = 1;
    uint64_t evidence_version = 0;
    uint32_t inflight_count = 0;
    uint32_t queued_actions = 0;
    uint32_t dirty_regions = 0;
    uint64_t total_admitted = 0;
    uint64_t total_completed = 0;
    uint64_t total_rejected_stale = 0;
    uint64_t total_rejected_budget = 0;
    uint64_t total_rejected_credits = 0;
    uint64_t total_rejected_unsupported = 0;
    uint64_t total_failed = 0;
    uint64_t total_dropped_telemetry = 0;
    uint64_t total_critical_backpressure = 0;
    double last_tick_us = 0.0;
    double available_budget_ms = 0.0;
    double last_event_to_map_correction_ms = 0.0;
    double max_event_to_map_correction_ms = 0.0;
    double mean_event_to_map_correction_ms = 0.0;
    uint64_t event_to_map_correction_samples = 0;
};

struct SchedulerConfig {
    size_t max_completions_per_tick = 16;
    size_t max_critical_evidence_per_tick = 32;
    size_t max_telemetry_evidence_per_tick = 32;
    size_t max_admissions_per_tick = 4;
    size_t max_pending_actions = 512;
    size_t critical_queue_capacity = 256;
    size_t telemetry_queue_capacity = 256;
    size_t max_regions = 4096;
    size_t trace_capacity = 4096;
    AdmissionConfig admission{};
};

struct TickResult {
    uint64_t tick_index = 0;
    uint32_t completions_processed = 0;
    uint32_t critical_evidence_processed = 0;
    uint32_t telemetry_evidence_processed = 0;
    uint32_t actions_admitted = 0;
    uint32_t stale_actions_pruned = 0;
    uint32_t budget_rejections = 0;
    uint32_t credit_rejections = 0;
    double tick_duration_us = 0.0;
};

// Abstract non-blocking executor interface consumed by AgoScheduler.
class ActionExecutorInterface {
public:
    virtual ~ActionExecutorInterface() = default;
    virtual SubmitResult try_submit(const AGOAction& action) noexcept = 0;
    virtual size_t poll_completed(CompletionRecord* out_buf, size_t max_count) noexcept = 0;
    virtual uint32_t available_credits() const noexcept = 0;
    virtual uint32_t inflight_count() const noexcept = 0;
    virtual double inflight_cost_ms() const noexcept = 0;
};

// Native C++ non-blocking AGO control-plane scheduler.
// All hot-path state is preallocated in the constructor; tick() is strictly noexcept,
// bounded, non-blocking, and performs zero heap allocations.
class AgoScheduler {
public:
    explicit AgoScheduler(
        const SchedulerConfig& config = {},
        ActionExecutorInterface* executor = nullptr
    );

    void set_executor(ActionExecutorInterface* executor) noexcept { executor_ = executor; }

    // Non-blocking producer APIs:
    // Critical evidence (pose revisions, contradictions, structural completions) NEVER drops;
    // returns false only on queue-full backpressure.
    bool push_critical_evidence(const EvidenceMessage& msg) noexcept;

    // Non-critical telemetry evidence drops oldest when full (returns true if an old item was dropped).
    bool push_telemetry_evidence(const EvidenceMessage& msg) noexcept;

    // Wait-free latest-wins deadline budget update.
    void publish_deadline_budget(const DeadlineBudget& budget) noexcept;

    // Enqueues a candidate AGOAction into the preallocated pending pool.
    // Returns false if pending pool is full or action is malformed.
    bool enqueue_action(const AGOAction& action) noexcept;

    // Main control-plane tick: strictly non-blocking, bounded, zero-allocation.
    TickResult tick() noexcept;

    // Non-blocking latest-wins summary consumer.
    bool try_consume_summary(SchedulerSummary& out) noexcept {
        return summary_mailbox_.try_consume(out);
    }

    const SchedulerSummary& latest_summary() const noexcept { return summary_; }
    const DirtyTracker& dirty_tracker() const noexcept { return dirty_tracker_; }
    const EventTrace& event_trace() const noexcept { return trace_; }

    uint64_t map_version() const noexcept { return map_version_; }
    uint64_t pose_version() const noexcept { return pose_version_; }
    uint64_t evidence_version() const noexcept { return evidence_version_; }

    void set_region_frozen(RegionId id, bool frozen) noexcept {
        dirty_tracker_.set_frozen(id, frozen, tick_count_);
    }

    void set_region_retracted(RegionId id, bool retracted) noexcept {
        dirty_tracker_.set_retracted(id, retracted, tick_count_);
    }

    void bump_map_version(uint64_t delta = 1u) noexcept { map_version_ += delta; }
    void bump_pose_version(uint64_t delta = 1u) noexcept { pose_version_ += delta; }

private:
    void handle_evidence(const EvidenceMessage& msg, bool is_critical) noexcept;
    void handle_completion(const CompletionRecord& rec) noexcept;
    uint32_t prune_stale_actions() noexcept;

    SchedulerConfig config_{};
    ActionExecutorInterface* executor_ = nullptr;
    DirtyTracker dirty_tracker_;
    AdmissionController admission_;
    EventTrace trace_;

    ReliableVersionedQueue<EvidenceMessage> critical_evidence_queue_;
    BoundedDropOldestQueue<EvidenceMessage> telemetry_evidence_queue_;
    LatestWinsMailbox<DeadlineBudget> budget_mailbox_;
    LatestWinsMailbox<SchedulerSummary> summary_mailbox_;

    // Preallocated pending action array (fixed capacity, swap-and-pop removal -> zero alloc)
    std::vector<AGOAction> pending_actions_;
    size_t pending_count_ = 0;

    // Preallocated scratch buffer for executor completion polling
    std::vector<CompletionRecord> completion_scratch_;

    DeadlineBudget current_budget_{};
    SchedulerSummary summary_{};

    uint64_t tick_count_ = 0;
    uint64_t map_version_ = 1;
    uint64_t pose_version_ = 1;
    uint64_t evidence_version_ = 0;
    uint64_t ticks_since_last_maintenance_ = 0;
    ActionId next_action_id_ = 1;
    double event_to_map_correction_sum_ms_ = 0.0;
};

}  // namespace vksplat::ago
