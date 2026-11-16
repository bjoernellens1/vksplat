// Per-step GPU gate for batched training steps.
//
// Runs once per step after the tile-intersection prefix sum. It reads the intersection count n on the GPU,
// checks it against the capacity of the key/value/sort buffers, and writes
//   * isect_count[0]  : n (0 when the step must not run), read by the radix sort and tile-range kernels,
//   * args            : VkDispatchIndirectCommand per gated dispatch slot (all zero when the step must not run),
//   * status          : [0] steps completed, [1] sticky overflow flag, [2] n of the first overflowing step,
//                       [4+step] n per step (copied to the host once, after the batch's fence).
// A step after an overflow sees the sticky flag and is skipped as well, so an overflowing step leaves the
// parameters and optimizer state untouched and the host can grow the buffers and redo it exactly.

static const uint32_t NUM_SLOTS = 8;
static const uint32_t SLOT_SORT = 1;         // radix sort upsweep/downsweep: ceil(n / PARTITION_SIZE)
static const uint32_t SLOT_TILE_RANGES = 2;  // compute_tile_ranges: ceil((n + 1) / 256)
static const uint32_t PARTITION_SIZE = 4096;

struct GateUniforms {
    uint32_t num_splats;
    uint32_t capacity;
    uint32_t step_idx;
    uint32_t _pad;
    uint4 base[NUM_SLOTS];  // dispatch group counts of the host-known slots
};

layout(binding=0) StructuredBuffer<int32_t> index_buffer_offset;  // inclusive prefix sum of tiles_touched
layout(binding=1) RWStructuredBuffer<uint32_t> status;
layout(binding=2) RWStructuredBuffer<uint32_t> isect_count;
layout(binding=3) RWStructuredBuffer<uint32_t> args;

[numthreads(1, 1, 1)]
void main(uniform GateUniforms u) {
    uint32_t n = (uint32_t)max(index_buffer_offset[u.num_splats - 1], 0);
    bool was_ok = status[1] == 0;
    bool ok = was_ok && n <= u.capacity;
    if (was_ok && !ok) {
        status[1] = 1;
        status[2] = n;
    }
    if (ok)
        status[0] = u.step_idx + 1;
    status[4 + u.step_idx] = n;
    isect_count[0] = ok ? n : 0;
    for (uint32_t s = 0; s < NUM_SLOTS; s++) {
        uint32_t o = (u.step_idx * NUM_SLOTS + s) * 4;
        uint4 b = ok ? u.base[s] : uint4(0);
        if (s == SLOT_SORT)
            b = ok ? uint4((n + PARTITION_SIZE - 1) / PARTITION_SIZE, 1, 1, 0) : uint4(0);
        if (s == SLOT_TILE_RANGES)
            b = ok ? uint4((n + 256) / 256, 1, 1, 0) : uint4(0);
        args[o] = b.x;
        args[o + 1] = b.y;
        args[o + 2] = b.z;
        args[o + 3] = 0;
    }
}
