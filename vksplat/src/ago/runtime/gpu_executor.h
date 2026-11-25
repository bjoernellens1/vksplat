#pragma once

#include "../core/action_protocol.h"
#include "../core/scheduler.h"

#include <vulkan/vulkan.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vksplat::ago {

struct CompletionBatch {
    static constexpr size_t kMaxBatch = 16;
    std::array<CompletionRecord, kMaxBatch> records{};
    size_t count = 0;
};

// Non-blocking Vulkan GPU executor backed by a preallocated ring of command buffers and VkFences.
// Invariants:
//   - try_submit() NEVER blocks; returns SubmitResult::NO_CREDITS when all slots are in-flight.
//   - poll_completed() queries vkGetFenceStatus() ONLY; NEVER calls vkWaitForFences/vkQueueWaitIdle/vkDeviceWaitIdle.
class NonBlockingGpuExecutor : public ActionExecutorInterface {
public:
    static constexpr uint32_t kDefaultRingSlots = 4u;

    using RecordCallback = std::function<SubmitResult(VkCommandBuffer, const AGOAction&, uint32_t& recorded_ops)>;
    using TeardownDrainCallback = std::function<void()>;

    NonBlockingGpuExecutor() noexcept = default;
    ~NonBlockingGpuExecutor() override;

    NonBlockingGpuExecutor(const NonBlockingGpuExecutor&) = delete;
    NonBlockingGpuExecutor& operator=(const NonBlockingGpuExecutor&) = delete;

    // Allocates command pool, command buffers, and fences once before hot-path execution.
    bool init(
        VkDevice device,
        VkQueue queue,
        uint32_t queue_family_index,
        uint32_t ring_slots = kDefaultRingSlots
    ) noexcept;

    void set_record_callback(RecordCallback cb) { record_cb_ = std::move(cb); }
    void set_teardown_drain_callback(TeardownDrainCallback cb) { teardown_drain_cb_ = std::move(cb); }

    // Safely drains any in-flight submissions before destroying Vulkan command buffers and fences.
    // Intended for teardown only (outside the scheduler hot path).
    void shutdown(uint64_t drain_timeout_ms = 5000u) noexcept;

    bool is_initialized() const noexcept { return initialized_; }

    SubmitResult try_submit(const AGOAction& action) noexcept override;
    CompletionBatch poll_completed_batch() noexcept;
    size_t poll_completed(CompletionRecord* out_buf, size_t max_count) noexcept override;

    uint32_t available_credits() const noexcept override;
    uint32_t inflight_count() const noexcept override { return inflight_count_; }
    double inflight_cost_ms() const noexcept override { return inflight_cost_ms_; }
    uint32_t ring_capacity() const noexcept { return static_cast<uint32_t>(slots_.size()); }

private:
    struct CommandSlot {
        VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool in_use = false;
        AGOAction action{};
        std::chrono::steady_clock::time_point submitted_at{};
    };

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    std::vector<CommandSlot> slots_;
    size_t next_slot_hint_ = 0;
    uint32_t inflight_count_ = 0;
    double inflight_cost_ms_ = 0.0;
    bool initialized_ = false;
    RecordCallback record_cb_{};
    TeardownDrainCallback teardown_drain_cb_{};
};

// Deterministic CPU-only non-blocking executor for unit tests, fault injection, and latency benchmarks.
class SimulatedAgoExecutor : public ActionExecutorInterface {
public:
    explicit SimulatedAgoExecutor(uint32_t ring_slots = 4u, uint32_t default_latency_ticks = 1u);

    SubmitResult try_submit(const AGOAction& action) noexcept override;
    size_t poll_completed(CompletionRecord* out_buf, size_t max_count) noexcept override;

    uint32_t available_credits() const noexcept override;
    uint32_t inflight_count() const noexcept override { return inflight_count_; }
    double inflight_cost_ms() const noexcept override { return inflight_cost_ms_; }

    // Advances simulated GPU clock by `ticks` (completes slots whose countdown reaches 0).
    void advance_ticks(uint32_t ticks = 1u) noexcept;

    // Fault-injection controls
    void set_stalled(bool stalled) noexcept { stalled_ = stalled; }
    void set_fail_next_completion(bool fail) noexcept { fail_next_completion_ = fail; }
    void set_unsupported_op_mask(uint32_t mask) noexcept { unsupported_op_mask_ = mask; }
    void set_default_latency_ticks(uint32_t ticks) noexcept { default_latency_ticks_ = ticks; }

    uint64_t total_submitted() const noexcept { return total_submitted_; }
    uint64_t total_completed() const noexcept { return total_completed_; }
    const std::vector<ActionId>& submitted_order() const noexcept { return submitted_order_; }
    void clear_history() noexcept { submitted_order_.clear(); }

private:
    struct SimSlot {
        bool in_use = false;
        uint32_t remaining_ticks = 0;
        AGOAction action{};
    };

    std::vector<SimSlot> slots_;
    uint32_t default_latency_ticks_ = 1u;
    uint32_t inflight_count_ = 0;
    double inflight_cost_ms_ = 0.0;
    bool stalled_ = false;
    bool fail_next_completion_ = false;
    uint32_t unsupported_op_mask_ = 0u;
    uint64_t total_submitted_ = 0;
    uint64_t total_completed_ = 0;
    std::vector<ActionId> submitted_order_;
};

}  // namespace vksplat::ago
