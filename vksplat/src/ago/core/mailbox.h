#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vksplat::ago {

// Wait-free triple-buffered latest-wins mailbox (SPSC, zero allocations after construction).
// Producers overwrite unread values without blocking; consumers always observe the latest published snapshot.
template <typename T>
class LatestWinsMailbox {
public:
    LatestWinsMailbox() noexcept = default;

    void publish(const T& value) noexcept {
        slots_[writer_idx_] = value;
        const uint8_t publish_state = static_cast<uint8_t>(writer_idx_ | kDirtyBit);
        const uint8_t prev = shared_state_.exchange(publish_state, std::memory_order_acq_rel);
        writer_idx_ = static_cast<uint8_t>(prev & kIdxMask);
        if ((prev & kDirtyBit) != 0u) {
            overwritten_count_.fetch_add(1u, std::memory_order_relaxed);
        }
        publish_count_.fetch_add(1u, std::memory_order_relaxed);
    }

    bool try_consume(T& out) noexcept {
        const uint8_t cur = shared_state_.load(std::memory_order_acquire);
        if ((cur & kDirtyBit) == 0u) {
            return false;
        }
        const uint8_t prev = shared_state_.exchange(reader_idx_, std::memory_order_acq_rel);
        reader_idx_ = static_cast<uint8_t>(prev & kIdxMask);
        out = slots_[reader_idx_];
        consume_count_.fetch_add(1u, std::memory_order_relaxed);
        return true;
    }

    bool has_unread() const noexcept {
        return (shared_state_.load(std::memory_order_acquire) & kDirtyBit) != 0u;
    }

    uint64_t publish_count() const noexcept {
        return publish_count_.load(std::memory_order_relaxed);
    }

    uint64_t consume_count() const noexcept {
        return consume_count_.load(std::memory_order_relaxed);
    }

    uint64_t overwritten_count() const noexcept {
        return overwritten_count_.load(std::memory_order_relaxed);
    }

private:
    static constexpr uint8_t kIdxMask  = 0x03u;
    static constexpr uint8_t kDirtyBit = 0x04u;

    std::array<T, 3> slots_{};
    uint8_t writer_idx_ = 0u;
    uint8_t reader_idx_ = 2u;
    std::atomic<uint8_t> shared_state_{1u};  // middle slot initially clean
    std::atomic<uint64_t> publish_count_{0u};
    std::atomic<uint64_t> consume_count_{0u};
    std::atomic<uint64_t> overwritten_count_{0u};
};

// Bounded non-blocking ring queue that drops the oldest item when full.
// Preallocates fixed capacity at construction; zero hot-path heap allocations.
template <typename T>
class BoundedDropOldestQueue {
public:
    explicit BoundedDropOldestQueue(size_t capacity = 256)
        : capacity_(capacity > 0 ? capacity : 1),
          ring_(capacity_) {}

    // Never blocks on fullness; if full, drops the oldest entry and inserts `item`.
    // Returns true if an older entry was dropped.
    bool push(const T& item) noexcept {
        acquire_flag();
        bool dropped = false;
        if (count_ == capacity_) {
            head_ = (head_ + 1u) % capacity_;
            --count_;
            ++dropped_count_;
            dropped = true;
        }
        ring_[tail_] = item;
        tail_ = (tail_ + 1u) % capacity_;
        ++count_;
        ++pushed_count_;
        release_flag();
        return dropped;
    }

    bool try_pop(T& out) noexcept {
        acquire_flag();
        if (count_ == 0u) {
            release_flag();
            return false;
        }
        out = ring_[head_];
        head_ = (head_ + 1u) % capacity_;
        --count_;
        release_flag();
        return true;
    }

    template <typename Fn>
    size_t drain_up_to(size_t max_count, Fn&& fn) noexcept {
        size_t drained = 0u;
        T item{};
        while (drained < max_count && try_pop(item)) {
            fn(item);
            ++drained;
        }
        return drained;
    }

    size_t size() const noexcept {
        acquire_flag();
        const size_t n = count_;
        release_flag();
        return n;
    }

    size_t capacity() const noexcept { return capacity_; }

    bool empty() const noexcept { return size() == 0u; }

    uint64_t dropped_count() const noexcept {
        acquire_flag();
        const uint64_t d = dropped_count_;
        release_flag();
        return d;
    }

    uint64_t pushed_count() const noexcept {
        acquire_flag();
        const uint64_t p = pushed_count_;
        release_flag();
        return p;
    }

private:
    void acquire_flag() const noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Short bounded spin; critical section is O(1) memcpy.
        }
    }

    void release_flag() const noexcept {
        flag_.clear(std::memory_order_release);
    }

    size_t capacity_ = 1;
    std::vector<T> ring_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
    uint64_t dropped_count_ = 0;
    uint64_t pushed_count_ = 0;
    mutable std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// Lock-free bounded FIFO queue (Vyukov MPMC ring) that NEVER drops critical messages
// (pose revisions, contradictions, structural completions).
// When full, `try_push` returns false (explicit backpressure) rather than dropping.
template <typename T>
class ReliableVersionedQueue {
public:
    struct Entry {
        T payload{};
        uint64_t version = 0;
    };

    explicit ReliableVersionedQueue(size_t min_capacity = 256) {
        size_t cap = 2;
        while (cap < min_capacity) {
            cap <<= 1u;
        }
        capacity_ = cap;
        mask_ = cap - 1u;
        slots_ = std::vector<Slot>(cap);
        for (size_t i = 0; i < cap; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    // Non-blocking enqueue. Returns false if full (NEVER drops).
    bool try_push(const T& item, uint64_t version) noexcept {
        Slot* slot = nullptr;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            slot = &slots_[pos & mask_];
            const size_t seq = slot->seq.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1u, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                rejected_full_count_.fetch_add(1u, std::memory_order_relaxed);
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        slot->entry.payload = item;
        slot->entry.version = version;
        slot->seq.store(pos + 1u, std::memory_order_release);

        const size_t approx_size = enqueue_pos_.load(std::memory_order_relaxed) -
                                   dequeue_pos_.load(std::memory_order_relaxed);
        size_t hw = high_water_mark_.load(std::memory_order_relaxed);
        while (approx_size > hw && !high_water_mark_.compare_exchange_weak(hw, approx_size, std::memory_order_relaxed)) {
        }
        update_last_version(version);
        return true;
    }

    bool try_pop(T& out, uint64_t* out_version = nullptr) noexcept {
        Slot* slot = nullptr;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            slot = &slots_[pos & mask_];
            const size_t seq = slot->seq.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1u);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1u, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        out = slot->entry.payload;
        if (out_version) {
            *out_version = slot->entry.version;
        }
        slot->seq.store(pos + mask_ + 1u, std::memory_order_release);
        return true;
    }

    template <typename Fn>
    size_t drain_up_to(size_t max_count, Fn&& fn) noexcept {
        size_t drained = 0u;
        T item{};
        uint64_t ver = 0u;
        while (drained < max_count && try_pop(item, &ver)) {
            fn(item, ver);
            ++drained;
        }
        return drained;
    }

    size_t size() const noexcept {
        const size_t enq = enqueue_pos_.load(std::memory_order_acquire);
        const size_t deq = dequeue_pos_.load(std::memory_order_acquire);
        return (enq >= deq) ? (enq - deq) : 0u;
    }

    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size() == 0u; }
    size_t high_water_mark() const noexcept { return high_water_mark_.load(std::memory_order_relaxed); }
    uint64_t rejected_full_count() const noexcept { return rejected_full_count_.load(std::memory_order_relaxed); }
    uint64_t last_version() const noexcept { return last_version_.load(std::memory_order_relaxed); }

private:
    struct Slot {
        std::atomic<size_t> seq{0};
        Entry entry{};
        Slot() = default;
        Slot(const Slot& o) noexcept : seq(o.seq.load(std::memory_order_relaxed)), entry(o.entry) {}
        Slot& operator=(const Slot& o) noexcept {
            seq.store(o.seq.load(std::memory_order_relaxed), std::memory_order_relaxed);
            entry = o.entry;
            return *this;
        }
    };

    void update_last_version(uint64_t ver) noexcept {
        uint64_t cur = last_version_.load(std::memory_order_relaxed);
        while (ver > cur && !last_version_.compare_exchange_weak(cur, ver, std::memory_order_relaxed)) {
        }
    }

    size_t capacity_ = 0;
    size_t mask_ = 0;
    std::vector<Slot> slots_;
    std::atomic<size_t> enqueue_pos_{0};
    std::atomic<size_t> dequeue_pos_{0};
    std::atomic<size_t> high_water_mark_{0};
    std::atomic<uint64_t> rejected_full_count_{0};
    std::atomic<uint64_t> last_version_{0};
};

}  // namespace vksplat::ago
