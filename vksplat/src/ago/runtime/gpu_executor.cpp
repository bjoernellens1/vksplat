#include "gpu_executor.h"

#include <algorithm>
#include <thread>

namespace vksplat::ago {

namespace {
double steady_now_ms() noexcept {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
}  // namespace

NonBlockingGpuExecutor::~NonBlockingGpuExecutor() {
    shutdown();
}

bool NonBlockingGpuExecutor::init(
    VkDevice device,
    VkQueue queue,
    uint32_t queue_family_index,
    uint32_t ring_slots
) noexcept {
    shutdown();
    if (device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) {
        return false;
    }

    const uint32_t n_slots = std::max<uint32_t>(ring_slots, 1u);
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;

    if (vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool_) != VK_SUCCESS) {
        cmd_pool_ = VK_NULL_HANDLE;
        return false;
    }

    std::vector<VkCommandBuffer> raw_cmds(n_slots, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = n_slots;

    if (vkAllocateCommandBuffers(device, &alloc_info, raw_cmds.data()) != VK_SUCCESS) {
        vkDestroyCommandPool(device, cmd_pool_, nullptr);
        cmd_pool_ = VK_NULL_HANDLE;
        return false;
    }

    slots_.assign(n_slots, CommandSlot{});
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < n_slots; ++i) {
        slots_[i].cmd_buf = raw_cmds[i];
        if (vkCreateFence(device, &fence_info, nullptr, &slots_[i].fence) != VK_SUCCESS) {
            device_ = device;
            shutdown();
            return false;
        }
    }

    device_ = device;
    queue_ = queue;
    next_slot_hint_ = 0;
    inflight_count_ = 0;
    inflight_cost_ms_ = 0.0;
    initialized_ = true;
    return true;
}

void NonBlockingGpuExecutor::shutdown(uint64_t drain_timeout_ms) noexcept {
    if (device_ != VK_NULL_HANDLE) {
        // Teardown-only drain: ensure all in-flight submissions complete before destroying
        // command buffers and fences. Never called from the scheduler hot path.
        if (inflight_count_ > 0u) {
            if (teardown_drain_cb_) {
                try {
                    teardown_drain_cb_();
                } catch (...) {
                }
            }
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(drain_timeout_ms);
            while (inflight_count_ > 0u) {
                for (auto& slot : slots_) {
                    if (!slot.in_use || slot.fence == VK_NULL_HANDLE) {
                        continue;
                    }
                    const VkResult status = vkGetFenceStatus(device_, slot.fence);
                    if (status != VK_NOT_READY) {
                        slot.in_use = false;
                        if (inflight_count_ > 0u) {
                            --inflight_count_;
                        }
                    }
                }
                if (inflight_count_ == 0u || std::chrono::steady_clock::now() >= deadline) {
                    break;
                }
                std::this_thread::yield();
            }
        }

        for (auto& slot : slots_) {
            if (slot.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device_, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            slot.cmd_buf = VK_NULL_HANDLE;
            slot.in_use = false;
        }
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, cmd_pool_, nullptr);
            cmd_pool_ = VK_NULL_HANDLE;
        }
    }
    slots_.clear();
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    inflight_count_ = 0;
    inflight_cost_ms_ = 0.0;
    initialized_ = false;
}

uint32_t NonBlockingGpuExecutor::available_credits() const noexcept {
    if (!initialized_) {
        return 0u;
    }
    const uint32_t cap = static_cast<uint32_t>(slots_.size());
    return (cap > inflight_count_) ? (cap - inflight_count_) : 0u;
}

SubmitResult NonBlockingGpuExecutor::try_submit(const AGOAction& action) noexcept {
    if (!record_cb_) {
        return SubmitResult::REJECTED_NO_EXECUTOR;
    }
    if (!initialized_ || device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE) {
        return SubmitResult::DEVICE_ERROR;
    }

    const size_t n = slots_.size();
    CommandSlot* chosen = nullptr;
    for (size_t i = 0; i < n; ++i) {
        const size_t idx = (next_slot_hint_ + i) % n;
        if (!slots_[idx].in_use) {
            chosen = &slots_[idx];
            next_slot_hint_ = (idx + 1u) % n;
            break;
        }
    }
    if (!chosen) {
        return SubmitResult::NO_CREDITS;
    }

    if (vkResetFences(device_, 1, &chosen->fence) != VK_SUCCESS) {
        return SubmitResult::DEVICE_ERROR;
    }
    if (vkResetCommandBuffer(chosen->cmd_buf, 0) != VK_SUCCESS) {
        return SubmitResult::DEVICE_ERROR;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(chosen->cmd_buf, &begin_info) != VK_SUCCESS) {
        return SubmitResult::DEVICE_ERROR;
    }

    uint32_t recorded_ops = 0u;
    const SubmitResult rec_res = record_cb_(chosen->cmd_buf, action, recorded_ops);
    if (rec_res != SubmitResult::OK) {
        vkEndCommandBuffer(chosen->cmd_buf);
        return rec_res;
    }
    if (recorded_ops == 0u) {
        vkEndCommandBuffer(chosen->cmd_buf);
        return SubmitResult::REJECTED_NO_EXECUTOR;
    }

    if (vkEndCommandBuffer(chosen->cmd_buf) != VK_SUCCESS) {
        return SubmitResult::DEVICE_ERROR;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &chosen->cmd_buf;

    if (vkQueueSubmit(queue_, 1, &submit_info, chosen->fence) != VK_SUCCESS) {
        return SubmitResult::DEVICE_ERROR;
    }

    chosen->in_use = true;
    chosen->action = action;
    chosen->submitted_at = std::chrono::steady_clock::now();
    ++inflight_count_;
    inflight_cost_ms_ += static_cast<double>(action.estimated_cost_ms);
    return SubmitResult::OK;
}

CompletionBatch NonBlockingGpuExecutor::poll_completed_batch() noexcept {
    CompletionBatch batch{};
    batch.count = poll_completed(batch.records.data(), CompletionBatch::kMaxBatch);
    return batch;
}

size_t NonBlockingGpuExecutor::poll_completed(CompletionRecord* out_buf, size_t max_count) noexcept {
    if (!initialized_ || !out_buf || max_count == 0u || inflight_count_ == 0u) {
        return 0u;
    }

    const auto now = std::chrono::steady_clock::now();
    const double now_ms = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
    size_t completed = 0u;
    for (auto& slot : slots_) {
        if (!slot.in_use || completed >= max_count) {
            continue;
        }
        // Strictly non-blocking status query only.
        const VkResult status = vkGetFenceStatus(device_, slot.fence);
        if (status == VK_NOT_READY) {
            continue;
        }

        CompletionRecord rec{};
        rec.action_id = slot.action.action_id;
        rec.region_id = slot.action.region_id;
        rec.op = slot.action.op;
        rec.service_class = slot.action.service_class;
        rec.status = (status == VK_SUCCESS) ? CompletionStatus::OK : CompletionStatus::FAILED_EXECUTION;
        rec.actual_cost_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(now - slot.submitted_at).count()
        );
        rec.completed_at_ms = now_ms;
        if (slot.action.evidence_ingested_at_ms > 0.0) {
            rec.event_to_map_correction_ms = std::max(0.0, now_ms - slot.action.evidence_ingested_at_ms);
        } else if (slot.action.enqueued_at_ms > 0.0) {
            rec.event_to_map_correction_ms = std::max(0.0, now_ms - slot.action.enqueued_at_ms);
        } else {
            rec.event_to_map_correction_ms = static_cast<double>(rec.actual_cost_ms);
        }
        rec.was_exclusive = slot.action.requires_exclusive_region;

        slot.in_use = false;
        if (inflight_count_ > 0u) {
            --inflight_count_;
        }
        inflight_cost_ms_ = std::max(0.0, inflight_cost_ms_ - static_cast<double>(slot.action.estimated_cost_ms));

        out_buf[completed++] = rec;
    }
    return completed;
}

// ---------------------------------------------------------------------------
// SimulatedAgoExecutor
// ---------------------------------------------------------------------------

SimulatedAgoExecutor::SimulatedAgoExecutor(uint32_t ring_slots, uint32_t default_latency_ticks)
    : slots_(std::max<uint32_t>(ring_slots, 1u)),
      default_latency_ticks_(default_latency_ticks) {
    submitted_order_.reserve(4096u);
}

uint32_t SimulatedAgoExecutor::available_credits() const noexcept {
    const uint32_t cap = static_cast<uint32_t>(slots_.size());
    return (cap > inflight_count_) ? (cap - inflight_count_) : 0u;
}

SubmitResult SimulatedAgoExecutor::try_submit(const AGOAction& action) noexcept {
    const uint32_t op_bit = 1u << static_cast<uint32_t>(action.op);
    if ((unsupported_op_mask_ & op_bit) != 0u) {
        return SubmitResult::UNSUPPORTED_OP;
    }

    for (auto& slot : slots_) {
        if (!slot.in_use) {
            slot.in_use = true;
            slot.remaining_ticks = default_latency_ticks_;
            slot.action = action;
            ++inflight_count_;
            inflight_cost_ms_ += static_cast<double>(action.estimated_cost_ms);
            ++total_submitted_;
            if (submitted_order_.size() < submitted_order_.capacity()) {
                submitted_order_.push_back(action.action_id);
            }
            return SubmitResult::OK;
        }
    }
    return SubmitResult::NO_CREDITS;
}

void SimulatedAgoExecutor::advance_ticks(uint32_t ticks) noexcept {
    if (stalled_ || ticks == 0u) {
        return;
    }
    for (auto& slot : slots_) {
        if (slot.in_use) {
            slot.remaining_ticks = (slot.remaining_ticks > ticks) ? (slot.remaining_ticks - ticks) : 0u;
        }
    }
}

size_t SimulatedAgoExecutor::poll_completed(CompletionRecord* out_buf, size_t max_count) noexcept {
    if (!out_buf || max_count == 0u || inflight_count_ == 0u || stalled_) {
        return 0u;
    }
    size_t completed = 0u;
    for (auto& slot : slots_) {
        if (!slot.in_use || slot.remaining_ticks > 0u || completed >= max_count) {
            continue;
        }
        CompletionRecord rec{};
        rec.action_id = slot.action.action_id;
        rec.region_id = slot.action.region_id;
        rec.op = slot.action.op;
        rec.service_class = slot.action.service_class;
        if (fail_next_completion_) {
            rec.status = CompletionStatus::FAILED_EXECUTION;
            fail_next_completion_ = false;
        } else {
            rec.status = CompletionStatus::OK;
        }
        rec.realized_gain = (rec.status == CompletionStatus::OK) ? slot.action.expected_gain : 0.0f;
        rec.actual_cost_ms = slot.action.estimated_cost_ms;
        const double now_ms = steady_now_ms();
        rec.completed_at_ms = now_ms;
        if (slot.action.evidence_ingested_at_ms > 0.0) {
            rec.event_to_map_correction_ms = std::max(0.0, now_ms - slot.action.evidence_ingested_at_ms);
        } else if (slot.action.enqueued_at_ms > 0.0) {
            rec.event_to_map_correction_ms = std::max(0.0, now_ms - slot.action.enqueued_at_ms);
        } else {
            rec.event_to_map_correction_ms = static_cast<double>(rec.actual_cost_ms);
        }
        rec.was_exclusive = slot.action.requires_exclusive_region;

        slot.in_use = false;
        if (inflight_count_ > 0u) {
            --inflight_count_;
        }
        inflight_cost_ms_ = std::max(0.0, inflight_cost_ms_ - static_cast<double>(slot.action.estimated_cost_ms));
        ++total_completed_;

        out_buf[completed++] = rec;
    }
    return completed;
}

}  // namespace vksplat::ago
