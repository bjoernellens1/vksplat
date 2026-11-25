// GPU-native 1-workgroup (256-thread) AGO histogram cutoff & indirect dispatch shader.
//
// Reads `hist_buf` produced by `ago_gain_histogram`:
//   byte  0 (uint32): eligible_count
//   byte  4 (float):  total_predicted_gain
//   byte  8 (uint32): newborn_count
//   byte 12 (float):  max_predicted_gain
//   byte 16 + 4*b (b in 0..255):   mature bin counts
//   byte 16 + 4*(256+b) (b in 0..255): newborn bin counts
//
// Computes exact-K / budget cutoff & boundary quota on-device via shared-memory reverse
// prefix sum (bin 255 -> bin 0), and writes `cutoff_indirect_buf` (12 uint32 words = 48 bytes):
//   word  0 (uint32): VkDispatchIndirectCommand.x (groupCountX; 0 when EMPTY)
//   word  1 (uint32): VkDispatchIndirectCommand.y (1)
//   word  2 (uint32): VkDispatchIndirectCommand.z (1)
//   word  3 (uint32): cutoff_bin (0..256)
//   word  4 (uint32): boundary_quota
//   word  5 (uint32): selected_count (target_k)
//   word  6 (uint32): active_set_state (0=EMPTY, 1=PARTIAL, 2=FULL, 3=INVALID)
//   word  7 (uint32): selection_mode (0=EMPTY, 1=ALL_NEWBORN_PLUS_MATURE, 2=NEWBORN_ONLY)
//   word  8 (uint32): eligible_count
//   word  9 (uint32): newborn_count
//   word 10 (uint32): total_predicted_gain_bits
//   word 11 (uint32): max_predicted_gain_bits

#include "config.slang"

static const uint32_t BLOCK_SIZE = 256u;

static const uint32_t ACTIVE_SET_EMPTY   = 0u;
static const uint32_t ACTIVE_SET_PARTIAL = 1u;
static const uint32_t ACTIVE_SET_FULL    = 2u;
static const uint32_t ACTIVE_SET_INVALID = 3u;

struct AgoCutoffUniforms {
    uint32_t num_splats;
    uint32_t opt_block_size;
    float target_fraction;
    uint32_t max_budget_splats;  // 0 = unlimited; >0 caps target_k
};

layout(binding=0) ByteAddressBuffer hist_buf;
layout(binding=1) RWByteAddressBuffer cutoff_indirect_buf;

groupshared uint32_t s_scan[BLOCK_SIZE];
groupshared uint32_t s_cutoff_bin;
groupshared uint32_t s_boundary_quota;
groupshared uint32_t s_found;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uniform AgoCutoffUniforms u
) {
    const uint32_t lid = groupThreadID.x;

    // Load 16-byte histogram header
    const uint32_t raw_eligible = hist_buf.Load(0u);
    const uint32_t total_gain_bits = hist_buf.Load(4u);
    const uint32_t raw_newborn = hist_buf.Load(8u);
    const uint32_t max_gain_bits = hist_buf.Load(12u);

    const uint32_t eligible_cnt = min(raw_eligible, u.num_splats);
    const uint32_t newborn_cnt = min(raw_newborn, eligible_cnt);

    float frac = u.target_fraction;
    if (!isfinite(frac)) {
        frac = 0.0f;
    }
    frac = clamp(frac, 0.0f, 1.0f);

    uint32_t target_k = 0u;
    if (eligible_cnt > 0u && frac > 0.0f) {
        if (frac >= 1.0f) {
            target_k = eligible_cnt;
        } else {
            const float scaled = frac * float(eligible_cnt);
            const uint32_t rounded = uint32_t(floor(scaled + 0.5f));
            target_k = clamp(rounded, 1u, eligible_cnt);
        }
    }
    if (u.max_budget_splats > 0u && target_k > u.max_budget_splats) {
        target_k = u.max_budget_splats;
    }

    uint32_t active_state = ACTIVE_SET_EMPTY;
    uint32_t mode = 0u;
    uint32_t k_need = 0u;
    uint32_t hist_offset_bins = 0u;  // 0 = mature, 256 = newborn

    if (target_k == 0u || u.num_splats == 0u) {
        active_state = ACTIVE_SET_EMPTY;
        mode = 0u;
        k_need = 0u;
    } else if (target_k >= u.num_splats) {
        active_state = ACTIVE_SET_FULL;
        mode = 1u;
        k_need = 0u;
    } else {
        active_state = ACTIVE_SET_PARTIAL;
        if (newborn_cnt <= target_k) {
            mode = 1u;
            k_need = target_k - newborn_cnt;
            hist_offset_bins = 0u;
        } else {
            mode = 2u;
            k_need = target_k;
            hist_offset_bins = 256u;
        }
    }

    // Reverse bin index: lid == 0 -> bin 255 (highest gain), lid == 255 -> bin 0 (lowest gain)
    const uint32_t bin = 255u - lid;
    const uint32_t bin_count = (k_need > 0u)
        ? hist_buf.Load(16u + 4u * (hist_offset_bins + bin))
        : 0u;

    s_scan[lid] = bin_count;
    if (lid == 0u) {
        s_cutoff_bin = (active_state == ACTIVE_SET_FULL) ? 0u : 256u;
        s_boundary_quota = 0u;
        s_found = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // Inclusive Hillis-Steele prefix sum across all 256 bins (high gain -> low gain)
    [unroll]
    for (uint32_t offset = 1u; offset < BLOCK_SIZE; offset <<= 1u) {
        const uint32_t add_val = (lid >= offset) ? s_scan[lid - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_scan[lid] += add_val;
        GroupMemoryBarrierWithGroupSync();
    }

    if (k_need > 0u) {
        const uint32_t incl = s_scan[lid];
        const uint32_t excl = (lid > 0u) ? s_scan[lid - 1u] : 0u;
        if (excl < k_need && incl >= k_need) {
            s_cutoff_bin = bin;
            s_boundary_quota = k_need - excl;
            s_found = 1u;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u) {
        uint32_t out_cutoff_bin = s_cutoff_bin;
        uint32_t out_boundary_quota = s_boundary_quota;
        if (k_need > 0u && s_found == 0u) {
            // Defensive clamp if histogram total < k_need
            out_cutoff_bin = 0u;
            out_boundary_quota = 0u;
        }

        const uint32_t blk = max(u.opt_block_size, 1u);
        const uint32_t group_count_x = (active_state == ACTIVE_SET_EMPTY || target_k == 0u)
            ? 0u
            : ((target_k + blk - 1u) / blk);

        // VkDispatchIndirectCommand at offset 0
        cutoff_indirect_buf.Store(0u, group_count_x);
        cutoff_indirect_buf.Store(4u, 1u);
        cutoff_indirect_buf.Store(8u, 1u);

        // Cutoff & active-set state metadata
        cutoff_indirect_buf.Store(12u, out_cutoff_bin);
        cutoff_indirect_buf.Store(16u, out_boundary_quota);
        cutoff_indirect_buf.Store(20u, target_k);
        cutoff_indirect_buf.Store(24u, active_state);
        cutoff_indirect_buf.Store(28u, mode);
        cutoff_indirect_buf.Store(32u, eligible_cnt);
        cutoff_indirect_buf.Store(36u, newborn_cnt);
        cutoff_indirect_buf.Store(40u, total_gain_bits);
        cutoff_indirect_buf.Store(44u, max_gain_bits);
    }
}
