#pragma once

#include "../core/action_protocol.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vksplat::ago {

// Sparse RegionWorkState table keyed by stable RegionId.
// Preallocates a fixed-capacity open-addressing hash index, dense RegionWorkState pool,
// and bounded dirty ready-queue so hot-path updates are O(1), noexcept, zero-allocation,
// and never perform full-map scans.
class DirtyTracker {
public:
    explicit DirtyTracker(size_t max_regions = 4096);

    RegionWorkState* find(RegionId id) noexcept;
    const RegionWorkState* find(RegionId id) const noexcept;

    // Returns existing state or allocates from the preallocated pool.
    // Returns nullptr if the region table has reached max_regions capacity.
    RegionWorkState* get_or_create(RegionId id, uint64_t tick) noexcept;

    // Marks a region dirty and enqueues it into the dirty ready-queue (O(1), deduplicated).
    bool mark_dirty(
        RegionId id,
        uint32_t dirty_flags,
        float gain_delta,
        uint64_t map_version,
        uint64_t pose_version,
        uint64_t evidence_version,
        uint64_t tick
    ) noexcept;

    void clear_dirty(RegionId id, uint32_t flags_mask = 0xFFFFFFFFu) noexcept;
    void set_frozen(RegionId id, bool frozen, uint64_t tick) noexcept;
    void set_retracted(RegionId id, bool retracted, uint64_t tick) noexcept;

    void mark_inflight(RegionId id, bool exclusive, uint64_t tick) noexcept;
    void complete_inflight(RegionId id, bool exclusive, bool success, uint64_t tick) noexcept;

    // Drains up to `max_count` dirty RegionIds from the ready-queue without scanning all regions.
    size_t drain_dirty_regions(RegionId* out_ids, size_t max_count) noexcept;

    size_t region_count() const noexcept { return pool_used_; }
    size_t region_capacity() const noexcept { return pool_.size(); }
    size_t dirty_queue_size() const noexcept { return ready_count_; }
    uint64_t total_dirty_marks() const noexcept { return total_dirty_marks_; }
    uint64_t overflow_count() const noexcept { return overflow_count_; }

private:
    struct HashSlot {
        RegionId key = 0;
        uint32_t pool_idx = 0;
        bool occupied = false;
    };

    static size_t hash_region(RegionId id) noexcept {
        uint64_t x = id;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return static_cast<size_t>(x);
    }

    int32_t find_pool_index(RegionId id) const noexcept;
    void enqueue_ready(RegionWorkState& st) noexcept;

    std::vector<RegionWorkState> pool_;
    std::vector<HashSlot> table_;
    size_t table_mask_ = 0;
    size_t pool_used_ = 0;

    std::vector<RegionId> ready_ring_;
    size_t ready_head_ = 0;
    size_t ready_tail_ = 0;
    size_t ready_count_ = 0;

    uint64_t total_dirty_marks_ = 0;
    uint64_t overflow_count_ = 0;
};

}  // namespace vksplat::ago
