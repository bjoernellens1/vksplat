#include "dirty_tracking.h"

#include <algorithm>
#include <cmath>

namespace vksplat::ago {

DirtyTracker::DirtyTracker(size_t max_regions) {
    const size_t cap = std::max<size_t>(max_regions, 16u);
    pool_.resize(cap);
    ready_ring_.resize(cap);

    // Hash table sized to >= 2x capacity (power of two) for <= 0.5 load factor.
    size_t tcap = 32u;
    while (tcap < cap * 2u) {
        tcap <<= 1u;
    }
    table_.resize(tcap);
    table_mask_ = tcap - 1u;
}

int32_t DirtyTracker::find_pool_index(RegionId id) const noexcept {
    size_t idx = hash_region(id) & table_mask_;
    const size_t tsize = table_.size();
    for (size_t probe = 0; probe < tsize; ++probe) {
        const HashSlot& slot = table_[idx];
        if (!slot.occupied) {
            return -1;
        }
        if (slot.key == id) {
            return static_cast<int32_t>(slot.pool_idx);
        }
        idx = (idx + 1u) & table_mask_;
    }
    return -1;
}

RegionWorkState* DirtyTracker::find(RegionId id) noexcept {
    const int32_t pi = find_pool_index(id);
    return (pi >= 0) ? &pool_[static_cast<size_t>(pi)] : nullptr;
}

const RegionWorkState* DirtyTracker::find(RegionId id) const noexcept {
    const int32_t pi = find_pool_index(id);
    return (pi >= 0) ? &pool_[static_cast<size_t>(pi)] : nullptr;
}

RegionWorkState* DirtyTracker::get_or_create(RegionId id, uint64_t tick) noexcept {
    size_t idx = hash_region(id) & table_mask_;
    const size_t tsize = table_.size();
    for (size_t probe = 0; probe < tsize; ++probe) {
        HashSlot& slot = table_[idx];
        if (!slot.occupied) {
            if (pool_used_ >= pool_.size()) {
                ++overflow_count_;
                return nullptr;
            }
            const uint32_t pi = static_cast<uint32_t>(pool_used_++);
            slot.key = id;
            slot.pool_idx = pi;
            slot.occupied = true;

            RegionWorkState& st = pool_[pi];
            st = RegionWorkState{};
            st.region_id = id;
            st.last_updated_tick = tick;
            return &st;
        }
        if (slot.key == id) {
            return &pool_[slot.pool_idx];
        }
        idx = (idx + 1u) & table_mask_;
    }
    ++overflow_count_;
    return nullptr;
}

void DirtyTracker::enqueue_ready(RegionWorkState& st) noexcept {
    if (st.in_ready_queue) {
        return;
    }
    if (ready_count_ >= ready_ring_.size()) {
        return;
    }
    ready_ring_[ready_tail_] = st.region_id;
    ready_tail_ = (ready_tail_ + 1u) % ready_ring_.size();
    ++ready_count_;
    st.in_ready_queue = true;
}

bool DirtyTracker::mark_dirty(
    RegionId id,
    uint32_t dirty_flags,
    float gain_delta,
    uint64_t map_version,
    uint64_t pose_version,
    uint64_t evidence_version,
    uint64_t tick
) noexcept {
    RegionWorkState* st = get_or_create(id, tick);
    if (!st) {
        return false;
    }
    st->dirty_flags |= dirty_flags;
    if (std::isfinite(gain_delta) && gain_delta > 0.0f) {
        st->pending_gain += gain_delta;
    }
    if (map_version > st->map_version) {
        st->map_version = map_version;
    }
    if (pose_version > st->pose_version) {
        st->pose_version = pose_version;
    }
    if (evidence_version > st->evidence_version) {
        st->evidence_version = evidence_version;
    }
    st->last_updated_tick = tick;
    ++total_dirty_marks_;
    if (st->dirty_flags != DIRTY_NONE) {
        enqueue_ready(*st);
    }
    return true;
}

void DirtyTracker::clear_dirty(RegionId id, uint32_t flags_mask) noexcept {
    RegionWorkState* st = find(id);
    if (!st) {
        return;
    }
    st->dirty_flags &= ~flags_mask;
    if (st->dirty_flags == DIRTY_NONE) {
        st->pending_gain = 0.0f;
    }
}

void DirtyTracker::set_frozen(RegionId id, bool frozen, uint64_t tick) noexcept {
    RegionWorkState* st = get_or_create(id, tick);
    if (!st) {
        return;
    }
    st->frozen = frozen;
    st->last_updated_tick = tick;
}

void DirtyTracker::set_retracted(RegionId id, bool retracted, uint64_t tick) noexcept {
    RegionWorkState* st = get_or_create(id, tick);
    if (!st) {
        return;
    }
    st->retracted = retracted;
    st->last_updated_tick = tick;
}

void DirtyTracker::mark_inflight(RegionId id, bool exclusive, uint64_t tick) noexcept {
    RegionWorkState* st = get_or_create(id, tick);
    if (!st) {
        return;
    }
    ++st->inflight_ops;
    if (exclusive) {
        st->exclusive_locked = true;
    }
    st->last_updated_tick = tick;
}

void DirtyTracker::complete_inflight(RegionId id, bool exclusive, bool success, uint64_t tick) noexcept {
    RegionWorkState* st = find(id);
    if (!st) {
        return;
    }
    if (st->inflight_ops > 0u) {
        --st->inflight_ops;
    }
    if (exclusive) {
        st->exclusive_locked = false;
    }
    if (success) {
        ++st->completed_ops;
    }
    st->last_updated_tick = tick;
}

size_t DirtyTracker::drain_dirty_regions(RegionId* out_ids, size_t max_count) noexcept {
    if (!out_ids || max_count == 0u) {
        return 0u;
    }
    size_t written = 0u;
    while (written < max_count && ready_count_ > 0u) {
        const RegionId rid = ready_ring_[ready_head_];
        ready_head_ = (ready_head_ + 1u) % ready_ring_.size();
        --ready_count_;
        RegionWorkState* st = find(rid);
        if (st) {
            st->in_ready_queue = false;
            if (st->dirty_flags != DIRTY_NONE) {
                out_ids[written++] = rid;
            }
        }
    }
    return written;
}

}  // namespace vksplat::ago
